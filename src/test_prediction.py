from predict import predict


text = input("Enter your feelings : ")

result = predict(text)

print("\nPrediction :", result)