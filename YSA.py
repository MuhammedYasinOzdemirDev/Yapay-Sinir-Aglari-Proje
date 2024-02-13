from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# Assume that the tokenizer and model have already been fit and trained as per previous code

# Modeli yükle
model = load_model('best_model.keras')

# Maksimum uzunluk ve tokenizer ayarları
max_length = 964
tokenizer = Tokenizer(char_level=True)

# Veri setini yükleyerek tokenizer'ı eğit
def load_and_train_tokenizer(file_path):
    import pandas as pd

    try:
        df = pd.read_csv(file_path, header=None, quotechar='"')
        train_urls = df[1].tolist()
        tokenizer.fit_on_texts(train_urls)
    except pd.errors.ParserError as e:
        print(f'ParserError: {e}')

load_and_train_tokenizer('updated_train.csv')

def load_dataset(file_path):
    df = pd.read_csv(file_path, header=None, quotechar='"')
    return df

# Load test dataset
#test_df = load_dataset('dataset_phishing.csv')  # This should load your test data
#test_df = load_dataset('phishing_site_urls.csv')  # This should load your test data
test_df = load_dataset('combined_dataset.csv')  # This should load your test data


# Preprocess the URLs in the test dataset
test_urls = test_df.iloc[:, 0].tolist()  # URLs are assumed to be in the first column
test_labels = test_df.iloc[:, -1].tolist()  # Labels are assumed to be in the last column

# Encode the labels
label_encoder = LabelEncoder()
test_labels_encoded = label_encoder.fit_transform(test_labels)

# Tokenize and pad the test URLs
test_sequences = tokenizer.texts_to_sequences(test_urls)
test_data = pad_sequences(test_sequences, maxlen=max_length)

# Make predictions
predictions = model.predict(test_data)

# Evaluate the predictions
# Assuming binary classification ('legitimate' is 0 and 'phishing' is 1)
y_pred = (predictions <= 0.5).astype(int).flatten()  # Notice the <= instead of >
y_true = np.array(test_labels_encoded)

# Calculate accuracy
accuracy = np.mean(y_pred == y_true)
print(f"Model Accuracy: {accuracy:.2%}")