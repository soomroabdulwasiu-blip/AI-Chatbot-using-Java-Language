*AI Chatbot using Java Language*

A simple chatbot made in Java. It answers common questions (FAQs), understands small spelling mistakes, and can be used in a desktop window, a web page, or the console.

No extra libraries are needed. Only Java is required.

*Features*

Answers frequently asked questions about orders, shipping, returns, payments and more
Understands different ways of asking the same question
Fixes small spelling mistakes (for example, "pasword" becomes "password")
Replies to greetings, thanks, goodbye, time and date
Remembers your last question to understand short follow-ups
You can teach it new answers while chatting
Three ways to use it: desktop window, web page, console

*How It Works*

NLP (language processing): The bot breaks your sentence into words, removes common words like "the" and "is", and reduces words to their root (for example, "shipping" becomes "ship"). It also treats similar words as the same (for example, "parcel" and "package").
Machine learning logic: The bot turns every FAQ and your question into numbers using TF-IDF. It then finds the FAQ that is most similar to your question using cosine similarity.
Rule-based replies: Simple patterns handle greetings, thanks, goodbye, time and date.
Confidence: If the bot is not sure, it suggests similar questions. If it does not know, it says so.
Requirements
Java JDK 15 or newer

*How to Run*

Compile:

javac Chatbot.java

*Example Questions*

What are your opening hours?
Where is my order?
How long does shipping take?
How do I return an item?
Which payment methods do you accept?
I forgot my password
Do you ship internationally?
Is my payment information safe?
Teach the Bot New Answers

While chatting, type:

teach: Do you sell gift cards? => Yes, gift cards start from $10.

The bot saves this in faq.txt and remembers it next time.

You can also open faq.txt and add lines in this format, then restart the bot:

question => answer
Project Files
File	Purpose
Chatbot.java	The complete chatbot program
faq.txt	The questions and answers (created on first run)

*Author Name*
Abdul Wasiu