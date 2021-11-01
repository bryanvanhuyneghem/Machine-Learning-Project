import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('punkt')

# read in the cleaned file
df_selected = pd.read_pickle('corpus_balanced3_cleaned_scarce_elimination.pkl')

lemmatizer = WordNetLemmatizer()
i = 0



# Tokenize the complaint
for ind, complaint in df_selected["Consumer complaint narrative"].items():
    i = i+1
    words = nltk.word_tokenize(complaint)
    new_words = []
    # Lemmatize the words in the complaint
    for word in words:
        new_words.append(lemmatizer.lemmatize(word))
    s = ' '.join(new_words)
    #print(ind, "(old): ", df_selected["Consumer complaint narrative"][ind])
    df_selected["Consumer complaint narrative"][ind] = s
    #print(ind, ": ", s)
    #print("**********************************************************")
    if (i % 1000) == 0:
        print(i)
    # if i == 10:
    #     break

#print(df_selected["Consumer complaint narrative"])
print("nulls in df_selected:", df_selected["Consumer complaint narrative"].isnull().sum())

# if for some reason, some complaints are null; remove them, because there aren't very many of them anyway
df_bcl = df_selected.dropna()

# pickle for later use
df_selected.to_csv("corpus_balanced3_cleaned_scarce_elimination.csv", index=False)
