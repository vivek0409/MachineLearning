import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import nltk
nltk.download('punkt_tab')
from nltk import word_tokenize
import string, re
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download('wordnet')

sms_spam = pd.read_csv('../../TestData/sms_spam.csv')
sms_spam.columns.values[0]="Label"
sms_spam.columns.values[1]="Sms"
# print(sms_spam.shape)
# print(sms_spam.head())
# print(sms_spam.isnull().sum())

sms_spam = sms_spam.reset_index(drop=True)

# sms_spam["Label"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)
# plt.ylabel("Spam vs Ham")
# plt.legend(["Ham", "Spam"])
# plt.savefig("image.jpg")

sms_spam['num char'] = sms_spam['Sms'].apply(len)

sms_spam['num_words'] = sms_spam['Sms'].apply(lambda x : len(nltk.word_tokenize(x)))

sms_spam['num_sent'] =sms_spam['Sms'].apply(lambda x : len(nltk.sent_tokenize(x)))
# print(sms_spam.head())

port_stemmer = PorterStemmer()
lan_stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
# Create a function to generate cleaned data from raw text
def clean_text(text):
    text = word_tokenize(text) # Create tokens
    text= " ".join(text) # Join tokens
    text = [char for char in text if char not in string.punctuation] # Remove punctuations
    text = ''.join(text) # Join the leters
    text = [char for char in text if char not in re.findall(r"[0-9]", text)] # Remove Numbers
    text = ''.join(text) # Join the leters
    text = [word.lower() for word in text.split() if word.lower() not in set(stopwords.words('english'))] # Remove common english words (I, you, we,...)
    text = ' '.join(text) # Join the leters
    # text = list(map(lambda x: lan_stemmer.stem(x), text.split()))
    text = list(map(lambda x: port_stemmer.stem(x), text.split()))
    # text = list(map(lambda x: lemmatizer.lemmatize(x), text.split()))
    return " ".join(text)   # error word
sms_spam['Clean Email'] = sms_spam['Sms'].apply(clean_text)

# print(sms_spam.columns)

wc = WordCloud(width = 2000, height = 1000, min_font_size = 10, background_color = 'Black')


spam_ = wc.generate(sms_spam[sms_spam['Label']=='spam']['Clean Email'].str.cat(sep = " "))
plt.figure(figsize = (15,6))
plt.show(spam_)

ham_ = wc.generate(sms_spam[sms_spam['Label']=='ham']['Clean Email'].str.cat(sep = " "))
plt.figure(figsize = (15,6))
plt.show(ham_)