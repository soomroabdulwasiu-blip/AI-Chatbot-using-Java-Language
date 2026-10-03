import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.List;
import java.util.regex.*;

/**
 * AI FAQ Chatbot (Java, no external libraries)
 *
 * NLP pipeline : lowercase -> tokenize -> stop-word removal -> light stemming
 * ML logic     : TF-IDF vectors + cosine similarity (nearest-FAQ retrieval)
 * Rule-based   : regex intents (greeting, thanks, time, date, help, bye)
 * Training     : built-in FAQ list + live teaching via  "teach: question => answer"
 *                (learned answers are saved to learned_faq.txt and reloaded on start)
 * Interface    : Swing GUI (default) or console (--cli)
 *
 * Compile: javac Chatbot.java      Run: java Chatbot        (console: java Chatbot --cli)
 */
public class Chatbot {

    // ===================== 1. NLP =====================
    static class NLP {
        static final Set<String> STOP = new HashSet<>(Arrays.asList(
            "a","an","the","is","are","was","were","be","am","do","does","did","i","you","we","it","my","your",
            "me","to","of","in","on","at","for","and","or","can","could","would","will","should","how","what",
            "when","who","which","please","there","this","that","with","about","have","has","if","so"));

        static List<String> tokenize(String text) {
            List<String> out = new ArrayList<>();
            for (String w : text.toLowerCase().replaceAll("[^a-z0-9 ]", " ").split("\\s+")) {
                if (w.isEmpty() || STOP.contains(w)) continue;
                out.add(stem(w));
            }
            return out;
        }

        // Very light suffix-stripping stemmer: "shipping"/"shipped"/"ships" -> "ship"
        static String stem(String w) {
            if (w.length() <= 3) return w;
            if (w.endsWith("ies") && w.length() > 4) return w.substring(0, w.length() - 3) + "y";
            if (w.endsWith("ing") && w.length() > 5) w = undouble(w.substring(0, w.length() - 3));
            else if (w.endsWith("ed") && w.length() > 4) w = undouble(w.substring(0, w.length() - 2));
            else if (w.matches(".*(s|x|z|ch|sh)es") ) w = w.substring(0, w.length() - 2);
            else if (w.endsWith("s") && !w.endsWith("ss")) w = w.substring(0, w.length() - 1);
            return w;
        }
        private static String undouble(String w) {
            int n = w.length();
            return (n > 2 && w.charAt(n - 1) == w.charAt(n - 2) && "lsz".indexOf(w.charAt(n - 1)) < 0)
                    ? w.substring(0, n - 1) : w;
        }
    }

    // ===================== 2. Knowledge base + TF-IDF model =====================
    static class Entry {
        final String question, answer;
        Map<String, Double> vec = new HashMap<>();
        Entry(String q, String a) { question = q; answer = a; }
    }

    static class Match {
        final Entry entry; final double score;
        Match(Entry e, double s) { entry = e; score = s; }
    }

    static class Brain {
        final List<Entry> entries = new ArrayList<>();
        Map<String, Double> idf = new HashMap<>();

        void add(String q, String a) { entries.add(new Entry(q, a)); }

        /** "Training": compute IDF over all FAQs and a normalised TF-IDF vector for each. */
        void train() {
            Map<String, Integer> df = new HashMap<>();
            List<Map<String, Double>> tfs = new ArrayList<>();
            for (Entry e : entries) {
                Map<String, Double> tf = new HashMap<>();
                for (String t : NLP.tokenize(e.question)) tf.merge(t, 2.0, Double::sum); // question words count double
                for (String t : NLP.tokenize(e.answer))   tf.merge(t, 1.0, Double::sum);
                tfs.add(tf);
                for (String t : tf.keySet()) df.merge(t, 1, Integer::sum);
            }
            idf.clear();
            int n = entries.size();
            for (Map.Entry<String, Integer> d : df.entrySet())
                idf.put(d.getKey(), Math.log((n + 1.0) / (d.getValue() + 1.0)) + 1.0);
            for (int i = 0; i < n; i++) entries.get(i).vec = weigh(tfs.get(i));
        }

        private Map<String, Double> weigh(Map<String, Double> tf) {
            Map<String, Double> v = new HashMap<>();
            double norm = 0;
            for (Map.Entry<String, Double> t : tf.entrySet()) {
                Double w = idf.get(t.getKey());
                if (w == null) continue;               // unseen word: ignore
                double x = t.getValue() * w;
                v.put(t.getKey(), x);
                norm += x * x;
            }
            norm = Math.sqrt(norm);
            if (norm > 0) for (Map.Entry<String, Double> e : v.entrySet()) e.setValue(e.getValue() / norm);
            return v;
        }

        /** Returns the best FAQ match for the user's sentence (cosine similarity). */
        Match best(String input) {
            Map<String, Double> tf = new HashMap<>();
            for (String t : NLP.tokenize(input)) tf.merge(t, 1.0, Double::sum);
            Map<String, Double> q = weigh(tf);
            Match best = new Match(null, 0);
            for (Entry e : entries) {
                double dot = 0;
                for (Map.Entry<String, Double> t : q.entrySet()) {
                    Double w = e.vec.get(t.getKey());
                    if (w != null) dot += t.getValue() * w;
                }
                if (dot > best.score) best = new Match(e, dot);
            }
            return best;
        }
    }

    // ===================== 3. Chat engine (rules + ML + learning) =====================
    static class Engine {
        static final double CONFIDENT = 0.30, MAYBE = 0.15;
        static final Path LEARNED = Paths.get("learned_faq.txt");
        final Brain brain = new Brain();
        final List<Object[]> rules = new ArrayList<>(); // {Pattern, Supplier<String>}

        Engine() {
            loadFaqs();
            loadLearned();
            brain.train();
            rule("^(hi|hello|hey|salam|good (morning|afternoon|evening))\\b.*", () ->
                 "Hello! I'm ShopBot. Ask me about orders, shipping, returns, payments or your account.");
            rule(".*\\b(thanks|thank you|thx)\\b.*", () -> "You're welcome! Anything else I can help with?");
            rule(".*\\b(bye|goodbye|see you|quit|exit)\\b.*", () -> "Goodbye! Have a great day.");
            rule(".*\\b(what time is it|current time|time now)\\b.*", () ->
                 "It's " + LocalDateTime.now().format(DateTimeFormatter.ofPattern("hh:mm a")) + ".");
            rule(".*\\b(today'?s date|what day|current date|what is the date)\\b.*", () ->
                 "Today is " + LocalDateTime.now().format(DateTimeFormatter.ofPattern("EEEE, d MMMM yyyy")) + ".");
            rule(".*\\b(who are you|your name)\\b.*", () -> "I'm ShopBot, a Java FAQ chatbot using TF-IDF language matching.");
            rule(".*\\b(help|what can you do)\\b.*", () ->
                 "Try: \"Where is my order?\", \"How do I return an item?\", \"Which payment methods do you accept?\"\n"
               + "You can also teach me:  teach: your question => my answer");
        }

        private void rule(String regex, java.util.function.Supplier<String> reply) {
            rules.add(new Object[]{Pattern.compile(regex, Pattern.CASE_INSENSITIVE), reply});
        }

        @SuppressWarnings("unchecked")
        String reply(String input) {
            String text = input.trim();
            if (text.isEmpty()) return "Please type a question.";

            // Learning command
            if (text.toLowerCase().startsWith("teach:")) return learn(text.substring(6));

            // Rule-based intents (greeting only if the message is short, so "hi, where is my order" goes to FAQ)
            boolean shortMsg = text.split("\\s+").length <= 4;
            for (Object[] r : rules) {
                Pattern p = (Pattern) r[0];
                boolean isGreeting = p.pattern().startsWith("^(hi");
                if (p.matcher(text).matches() && (!isGreeting || shortMsg))
                    return ((java.util.function.Supplier<String>) r[1]).get();
            }

            // ML retrieval
            Match m = brain.best(text);
            if (m.entry != null && m.score >= CONFIDENT) return m.entry.answer;
            if (m.entry != null && m.score >= MAYBE)
                return "I'm not fully sure. Did you mean: \"" + m.entry.question + "\"?\n(" + m.entry.answer + ")";
            return "Sorry, I don't know that yet. You can teach me:  teach: question => answer";
        }

        private String learn(String body) {
            String[] parts = body.split("=>", 2);
            if (parts.length < 2 || parts[0].trim().isEmpty() || parts[1].trim().isEmpty())
                return "Format:  teach: your question => the answer";
            String q = parts[0].trim(), a = parts[1].trim();
            brain.add(q, a);
            brain.train();
            try {
                Files.write(LEARNED, (q + " => " + a + System.lineSeparator()).getBytes(StandardCharsets.UTF_8),
                        StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            } catch (IOException ignored) { }
            return "Thanks! I've learned that.";
        }

        private void loadLearned() {
            if (!Files.exists(LEARNED)) return;
            try {
                for (String line : Files.readAllLines(LEARNED, StandardCharsets.UTF_8)) {
                    String[] p = line.split("=>", 2);
                    if (p.length == 2) brain.add(p[0].trim(), p[1].trim());
                }
            } catch (IOException ignored) { }
        }

        // ---- Training data: frequently asked questions ----
        private void loadFaqs() {
            String[][] faq = {
                {"What are your opening hours?", "We're open Monday to Saturday, 9 AM to 9 PM. Online support is available 24/7."},
                {"Where is my order? How can I track my order or package?", "Open 'My Orders' in your account and click 'Track'. You'll also get tracking details by email."},
                {"How long does shipping take? What is the delivery time?", "Standard delivery takes 3-5 business days; express delivery takes 1-2 business days."},
                {"How much does shipping cost? Is delivery free?", "Shipping is free on orders above $50. Otherwise a flat $4.99 applies."},
                {"What is your return policy? How do I return an item?", "You can return unused items within 30 days. Start a return from 'My Orders' and print the prepaid label."},
                {"How do I get a refund?", "Refunds are issued to your original payment method within 5-7 business days after we receive the return."},
                {"Which payment methods do you accept?", "We accept Visa, MasterCard, PayPal, bank transfer and cash on delivery."},
                {"How do I reset my password? I forgot my password.", "Click 'Forgot password' on the login page and follow the email link to set a new one."},
                {"How can I create an account? How do I sign up or register?", "Click 'Sign Up', enter your email and a password, then confirm via the verification email."},
                {"How do I cancel my order?", "Orders can be cancelled before they ship: go to 'My Orders' and choose 'Cancel order'."},
                {"How can I contact customer support? What is your phone number or email?", "Email support@shopease.example or call +1-800-555-0199 during business hours. Live chat is open 24/7."},
                {"Do you ship internationally? Do you deliver abroad?", "Yes, we ship to over 40 countries. International delivery takes 7-14 business days."},
                {"Do you offer discounts or coupon codes?", "Subscribe to our newsletter for a 10% welcome coupon and seasonal sale alerts."},
                {"Is my personal data and payment information safe? Is the site secure?", "Yes. All payments are encrypted with SSL and we never store full card numbers."},
                {"My item arrived damaged or defective. What should I do?", "Sorry about that! Report it within 7 days from 'My Orders' with a photo and we'll replace it for free."},
                {"Do you have a warranty?", "Electronics carry a 1-year manufacturer warranty. Keep your invoice as proof of purchase."},
            };
            for (String[] f : faq) brain.add(f[0], f[1]);
        }
    }

    // ===================== 4. Swing GUI =====================
    static class ChatWindow extends JFrame {
        private final JTextArea log = new JTextArea();
        private final JTextField input = new JTextField();
        private final Engine engine;

        ChatWindow(Engine engine) {
            super("ShopBot - AI FAQ Chatbot");
            this.engine = engine;
            log.setEditable(false);
            log.setLineWrap(true);
            log.setWrapStyleWord(true);
            log.setFont(new Font("SansSerif", Font.PLAIN, 14));
            log.setMargin(new Insets(10, 10, 10, 10));

            JButton send = new JButton("Send");
            JPanel bottom = new JPanel(new BorderLayout(6, 6));
            bottom.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));
            bottom.add(input, BorderLayout.CENTER);
            bottom.add(send, BorderLayout.EAST);

            JLabel header = new JLabel("  ShopBot - ask me anything about orders, shipping, returns...");
            header.setOpaque(true);
            header.setBackground(new Color(30, 90, 160));
            header.setForeground(Color.WHITE);
            header.setFont(new Font("SansSerif", Font.BOLD, 14));
            header.setBorder(BorderFactory.createEmptyBorder(10, 6, 10, 6));

            add(header, BorderLayout.NORTH);
            add(new JScrollPane(log), BorderLayout.CENTER);
            add(bottom, BorderLayout.SOUTH);

            ActionListener onSend = e -> send();
            send.addActionListener(onSend);
            input.addActionListener(onSend);

            setSize(520, 620);
            setLocationRelativeTo(null);
            setDefaultCloseOperation(EXIT_ON_CLOSE);
            append("ShopBot", "Hi! Type a question (or 'help').");
            SwingUtilities.invokeLater(input::requestFocusInWindow);
        }

        private void send() {
            String q = input.getText().trim();
            if (q.isEmpty()) return;
            input.setText("");
            append("You", q);
            input.setEnabled(false);
            // short delay to simulate typing, keeps the UI responsive
            javax.swing.Timer t = new javax.swing.Timer(450, e -> {
                append("ShopBot", engine.reply(q));
                input.setEnabled(true);
                input.requestFocusInWindow();
            });
            t.setRepeats(false);
            t.start();
        }

        private void append(String who, String msg) {
            log.append(who + ":  " + msg + "\n\n");
            log.setCaretPosition(log.getDocument().getLength());
        }
    }

    // ===================== 5. Entry point =====================
    public static void main(String[] args) {
        Engine engine = new Engine();
        if (args.length > 0 && args[0].equals("--cli")) {
            Scanner sc = new Scanner(System.in);
            System.out.println("ShopBot: Hi! Ask me something (type 'exit' to quit).");
            while (sc.hasNextLine()) {
                String line = sc.nextLine();
                System.out.println("ShopBot: " + engine.reply(line));
                if (line.toLowerCase().matches(".*\\b(bye|exit|quit)\\b.*")) break;
            }
            return;
        }
        SwingUtilities.invokeLater(() -> new ChatWindow(engine).setVisible(true));
    }
}
