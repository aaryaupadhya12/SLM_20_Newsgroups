from Test import predict_text

test_texts = [
    "I love playing basketball and football!",
    "This new graphics card is amazing for gaming",
    "Jesus Christ is my savior and I believe in God",
    "The stock market crashed today, all investors lost money",
    "Python is the best programming language for machine learning"
    "Medical Vaccines of COVID-19 has started Circulation"
]

for i, text in enumerate(test_texts, 1):
    predicted_label, confidence = predict_text(text)
    print(f"Example {i}:")
    print(f"  Text: '{text}'")
    print(f"  Predicted: {predicted_label}")
    print(f"  Confidence: {confidence:.2%}\n")