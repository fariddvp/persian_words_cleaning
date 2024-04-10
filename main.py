import pandas as pd
import re
import nltk
import string
import nltk
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('persian')
nltk.download('punkt')


df = pd.read_csv('/home/farid/Documents/TAAV_vscode_prj/persian_words_cleaning/src/persian_dataset.csv')
print(df)

df = pd.DataFrame(df)
df = df.iloc[0:10000,:]

# Converting dataset to string
df = df['true text'].to_string() 
print(df)
print('Step 01 --->>> Converted dataframe to string.')


# Remove new lines,tabs,tabs and special tags
df = df.replace("\\n", " ")
df = df.replace("\\t", " ")
df = df.replace("\u200c", " ")
df = df.replace("\u200e", " ")
df = df.replace("\u200b", " ")
df = df.replace("\u200f", " ")
df = df.replace("\xad", " ")
df = re.sub(re.compile(r'\s+'), " ", df)
print(df)
print('Step 02 --->>> Removed new lines,tabs,tabs and special tags from datastring.')


# Remove unwanted digits from dataset
unwanted_digit = ['0','1','2','3','4','5','6','7','8','9',
                 '۰','۱','۲','۳','۴','۵','۶','۷','۸','۹']

for digit in unwanted_digit:
    df = df.replace(digit, "")

print(df)
print('Step 03 --->>> Removed unwanted digits from datastring.')


# Define a pattern to match punctuations
punctuation_pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))

# Remove punctuations from string
df = punctuation_pattern.sub('', df)

print(df)
print('Step 04 --->>> Removed punctuations from datastring.')


# Encodeand decode string to 'utf-8'
df = df.encode("utf-8", "ignore") 
df = df.decode()
print(df)


# Removed url's from string
df = re.sub(r'https?://[a-zA-Z0-9\.\/\-_?=;&]*', '', df)
print(df)
print('Step 05 --->>> Removed url tags from datastring.')


# Removed HTML's from string
df = re.sub(r'<[^>]+>', '', df)
print(df)
print('Step 06 --->>> Removed html tags from datastring.')


# Removed custom punctuations from string
unwanted_punc = ['د:','"',"'",'=','”','،','؛','»','؟','#','«','@','&','%','.',',',':','\\','$','^','<','>','!','?','{','}',';','\n','\t','(',')','[',']','/','*','+','#','\u200c','\ufeff','-','_','|']

for punc in unwanted_punc:
    df = df.replace(punc, "")

print(df)
print('Step 07 --->>> Removed custom punctuations from datastring.')



# Tokenization string words
# Download the necessary NLTK data for tokenization
nltk.download('punkt')

# Tokenize the Persian text
tokens = word_tokenize(df)

print(tokens)
print('Step 08 --->>> Tokenized string words.')




# Remove custom stop words
# Download the necessary NLTK data
nltk.download('stopwords')

# Manually download the Persian stopwords resource
nltk.download('persian')

persian_stopwords = ["و", "در", "به", "از", "که", "ولی", "اما", "است", "با", "ها",
                     "این", "را", "شده", "بود", "شد", "خواهد", "تا", "او",
                     "کند", "اند", "برای", "اگر", "می", "آن", "وی", "های", "ای"]

# Remove stopwords from the tokens
filtered_tokens = [word for word in tokens if word.lower() not in persian_stopwords]

# Join the filtered tokens back into a text
filtered_text = ' '.join(filtered_tokens)

# Print the filtered text

tokens = filtered_text.split()
print(tokens)
print('Step 09 --->>> Remove custom stop words from the tokens.')




# Convert string to dataframe and rename column
df_persian = pd.DataFrame(tokens)
df_persian.rename(columns={0:'Words'}, inplace=True)
print(df_persian)
print('Step 10 --->>> Converted tokens to dataframe.')




# Remove repeated characters from each token
# Define a pattern to match repeated characters (2 or more repetitions)
pattern = re.compile(r'(.)\1+')

df_persian['Reduction Words'] = [pattern.sub(r'\1', token) for token in df_persian['Words']]

print(df_persian)
print('Step 11 --->>> Redutioned repeated character from words in dataframe.')


df_persian.head(20)



# Remove non-meaning words from dataframe
# Function to check if a word is meaningful
def is_meaningful(word):
    return word not in persian_stopwords

meaningful_words = [word if is_meaningful(word) else 'Removed' for word in df_persian['Reduction Words']]

# Assign the list of meaningful words to the DataFrame column
df_persian['Meaningful Words'] = meaningful_words

print(df_persian)
print('Step 12 --->>> Remove non-meaning words from dataframe.')



df_persian[df_persian['Meaningful Words'] == 'Removed'].value_counts()



# Creater a custom dictionary
df_persian_test = df_persian.iloc[0:100,:]

# Manually create a list of Persian words to add to the dictionary
random_seed = 392
persian_dictionary_test = df_persian['Words'].sample(20,replace=False,random_state=random_seed)
persian_dictionary_test


# Initialize SpellChecker
spell_checker = SpellChecker()


# Add Persian words to the dictionary
spell_checker.word_frequency.load_words(persian_dictionary_test)

# Perform spelling correction for each token
corrected_words = []
for token in df_persian_test['Meaningful Words']:
    corrected_token = spell_checker.correction(token)
    
    # If correction is found, use it; otherwise, keep the original word
    corrected_words.append(corrected_token if corrected_token is not None else token)

# Assign corrected words to the DataFrame
df_persian_test['Correct Words'] = corrected_words
print('Step 14 --->>> Corrected spelling words by adding custom words to the dictionary in the DataFrame.')


PATH_csv = '/home/farid/Documents/TAAV_vscode_prj/persian_words_cleaning/src/Output_Words.csv'
# Export dataset to csv
df_persian_test.to_csv(PATH_csv, index=False)

# Export dataset to txt
PATH_txt = '/home/farid/Documents/TAAV_vscode_prj/persian_words_cleaning/src/Output_Words.txt'
with open(PATH_txt,'w+') as file:
    for word in df_persian_test['Correct Words']:
        file.write(word + '\n')

print('Step 15 --->>> Exported the DataFrame to csv and txt format.')


