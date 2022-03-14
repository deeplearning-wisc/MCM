from src.tclip.dataset import prepare_dataframe

def translate_text(target = 'zh', text = "hi there"):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    import six
    from google.cloud import translate_v2 as translate

    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    results = translate_client.translate(text, target_language=target)
    translated = []
    for result in results:
        translated.append(result["translatedText"])
        #print(u"Text: {}".format(result["input"]))
        print(u"Translation: {}".format(result["translatedText"]))
        # print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))

texts_src = [
    'Good morning. What can I do for you?',
    'Read aloud and underline the sentences about booking a flight.',
    'May I have your name and telephone number?',
    'A cat is chasing after a dog'
]

train_df, valid_df = prepare_dataframe(lang = 'en', captions_path = 'dataset/coco/captions/en/processed_captions_train2014.csv', train_only = False)

# translate_text(target = 'de', text = texts_src)

df_to_translate = train_df
from pygtrans import Translate

# valid_df = valid_df[:10]
translated = []
bz = 10000
client = Translate()
for i in range(0, len(df_to_translate), bz):
    texts_src = df_to_translate.caption[i:i+bz]
    print(len(texts_src))
    # translated.extend(list(test_src))
    texts = client.translate(list(texts_src), target = 'fr') #zh, de
    for text in texts:
        translated.append(text.translatedText) 

print('done')
df_to_translate['de_caption'] = translated 
df_to_translate = df_to_translate.drop(columns = ['caption'])
from pathlib import Path  
filepath = Path('test/processed_captions_train2014_fr.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True) 
#or: 
# import os  
# os.makedirs('folder/subfolder', exist_ok=True) 
df_to_translate.to_csv(filepath, index = False)  