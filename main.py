# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from urllib import request
from bs4 import BeautifulSoup
from selenium import webdriver

import sys

import selenium

import nltk
import os
import spacy
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager 
from nltk.corpus import stopwords
from selenium.common.exceptions import StaleElementReferenceException 
from nltk.corpus import wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize.regexp import RegexpTokenizer

import string
import requests
import json
import re
from collections import defaultdict
from scraper import wikiscrape
import scraper
"""import ssl 
try: _create_unverified_https_context = ssl._create_unverified_context 
except AttributeError: pass 
else: ssl._create_default_https_context = _create_unverified_https_context 
nltk.download()"""
from io import StringIO  # Python3
from BigramChunker import *
from nltk.corpus import conll2000



sentiment = SentimentIntensityAnalyzer()
tokenizer = RegexpTokenizer("\s+", gaps=True)
punct = string.punctuation
sw = stopwords.words("english")
nlp = spacy.load('en_core_web_sm')
sim = spacy.load("en_core_web_md")
stemmer = nltk.PorterStemmer()
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
chunker = BigramChunker(train_sents)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def get_reviews_lod(driver, link, polarity):
    driver.get(link)
    reviews_list = driver.find_element_by_id("cm_cr-review_list")
    reviews_divs = reviews_list.find_elements_by_css_selector("div")
    reviews_lod = []
    for div in reviews_divs:
        try: 
            if "review" == div.get_attribute("data-hook"):
                review_dict = {}
                name = div.find_element_by_class_name("a-profile-name").text
                review_dict["polarity"] = polarity
                review_dict["name"] = name
                all_as = div.find_elements_by_css_selector("a")
                for a in all_as:
                    if "out of 5 stars" in a.get_attribute("title"):
                        title = a.get_attribute("title")
                        stars = float(title[:title.index("out of 5 stars")].strip())
                        review_dict["stars"] = stars
                    if a.get_attribute("data-hook") == "review-title":
                        review_dict["title"] = a.text
                all_spans = div.find_elements_by_css_selector("span")
                for span in all_spans:
                    if span.get_attribute("data-hook") == "review-body":
                        review_dict["content"] = span.text
                    if span.get_attribute("data-hook") == "helpful-vote-statement":
                        helpful_statement = span.text
                        if "people" in helpful_statement:
                            helpfuls = helpful_statement[:helpful_statement.index("people")].strip()
                            if "," in helpfuls:
                                helpfuls = helpfuls.replace(",", "")
                            review_dict["helpful"] = int(helpfuls)
                        elif "person" in helpful_statement:
                            review_dict["helpful"] = 1
                reviews_lod.append(review_dict)
        except StaleElementReferenceException as Exception:
            print("stale element reference")
    button_ul = reviews_list.find_element_by_class_name("a-pagination")
    next_button = button_ul.find_element_by_class_name("a-last")
    try:
        next_page = next_button.find_element_by_css_selector("a").get_attribute("href")
        print(next_page)
        if isinstance(next_page, str):
            reviews_lod.extend(get_reviews_lod(driver, next_page, polarity))
        else:
            return reviews_lod
    except selenium.common.exceptions.NoSuchElementException:
        return reviews_lod
    return reviews_lod


def get_dict_def(word, word_dict):
    url = "https://od-api.oxforddictionaries.com:443/api/v2/entries/" + "en" + "/" + word.lower()
    r = requests.get(url, headers={"app_id": "79a74762", "app_key": "fbbda56ba258d085e49162ed80758d43"})
    json_dict = r.json()
    result_list = json_dict["results"]
    for result in result_list:
        for L_entry in result["lexicalEntries"]:
            entry_list = L_entry["entries"]
            for entry in entry_list:
                for sense in entry["senses"]:
                    for definition in sense["definitions"]:
                        word_dict[word].append(definition)


def get_wiki_def(word):
    old_stdout = sys.stdout
    new_stdout = StringIO()
    sys.stdout = new_stdout
    word_wiki = wikiscrape.wiki(word)
    output = new_stdout.getvalue()
    sys.stdout = old_stdout
    lines = output.split("\n")
    for i in range(len(lines)):
        if "Wikipedia page loaded successfully" in lines[i]:
            if i + 3 < len(lines) - 1:
                return nltk.sent_tokenize(lines[i + 3]), word_wiki.commonwords(50)
    return ("blank", "blank")


def makeHighlight(pos_tags, keywords):
    highlight_indices = []
    for i, w_pos in enumerate(pos_tags):
        w1, pos = w_pos
        if w1 in keywords:
            highlight_indices.append(i)
            for j in range(1, 4):
                if i - j > 0:
                    w2, pos2 = pos_tags[i - j]
                    polarity = sentiment.polarity_scores(w2)
                    if (pos2 == "ADJ" or pos2 == "ADV") and abs(polarity["compound"]) > 0.05:
                        highlight_indices.append(i - j)

                if i + j < len(pos_tags):
                    w2, pos2 = pos_tags[i + j]
                    polarity = sentiment.polarity_scores(w2)
                    if (pos2 == "ADJ" or pos2 == "ADV") and abs(polarity["compound"]) > 0.05:
                        highlight_indices.append(i + j)

        polarity = sentiment.polarity_scores(w1)
        polarity1 = sentiment.polarity_scores(stemmer.stem(w1).lower())
        if abs(polarity["pos"] - polarity["neg"]) > 0.3 or abs(abs(polarity1["pos"] - polarity1["neg"]) > 0.3):
            highlight_indices.append(i)

    return list(set(highlight_indices))


def toggleComments(features, comwords, reviews_lod):
    keywords_result = [0, 0]
    print(len(reviews_lod))
    count = 0
    for review in reviews_lod:
        if len(review) < 5:
            continue
        count += 1
        content = review["title"] + " " + review["content"]

        lines = nltk.sent_tokenize(content)
        added = False
        for feature_word in features:
            for i, line in enumerate(lines):
                delimiters = "[" + "\\".join(punct + "\"") + "]"
                line = ' '.join(w for w in re.split(delimiters, line) if w)
                content_words = nltk.word_tokenize(line)
                if len(feature_word.split()) > 0 and feature_word in line:
                    added = True
                else:
                    for content_word in content_words:
                        if feature_word.lower() == content_word.lower():
                            added = True
                            break
            if added:
                break
        if added:
            title_words = nltk.word_tokenize(review["title"])
            title_pos_tags = nltk.pos_tag(title_words, tagset="universal")
            title_highlight_indices = makeHighlight(title_pos_tags, comwords)
            content_words = nltk.word_tokenize(review["content"])
            content_pos_tags = nltk.pos_tag(content_words, tagset="universal")
            content_highlight_indices = makeHighlight(content_pos_tags, comwords)
            sent_polarity_scores = sentiment.polarity_scores(line)
            keywords_result[0] += review["stars"]
            keywords_result[1] += sent_polarity_scores["compound"]
            if line in review["title"]:
                index = ("title", i)
            else:
                index = ("review", i - len(nltk.sent_tokenize(review["title"])))
            keywords_result.append((review["stars"], sent_polarity_scores["compound"], review["title"],
                                    review["content"], title_highlight_indices, content_highlight_indices,
                                    index))
    if len(keywords_result) <= 2:
        print(features, count)
        return keywords_result
    keywords_result[0] = keywords_result[0]/(len(keywords_result) - 2)
    keywords_result[1] = keywords_result[1] /(len(keywords_result) - 2)
    return keywords_result
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    
    url = "https://www.amazon.com/Samsung-Jet-Stick-Power-Cordless-VS20T7536T5/dp/B087V67QC8/ref=sr_1_1_sspa?crid=2GYUTY9MTWGU3&dchild=1&keywords=dyson%2Bvacuum%2Bcleaner&qid=1628773289&sprefix=dyson%2Bva%2Caps%2C151&sr=8-1-spons&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUFZRE9GSk5VR0o4R1kmZW5jcnlwdGVkSWQ9QTA5NjQzMDkzTTk2WUs0QkJUOE4yJmVuY3J5cHRlZEFkSWQ9QTA5MDM0NjYxNUZRWjRLREgxNkZOJndpZGdldE5hbWU9c3BfYXRmJmFjdGlvbj1jbGlja1JlZGlyZWN0JmRvTm90TG9nQ2xpY2s9dHJ1ZQ&th=1"
    #os.chmod("/Users/ericlee/Desktop/amazon_crawler/venv/Scripts", 755) (amazon_crawler/venv/Scripts/)
    #driver = webdriver.Chrome(executable_path="/Users/ericlee/Downloads/chromedriver.exe")
    driver = webdriver.Chrome(ChromeDriverManager().install())

    driver.get(url)
    title = driver.title
    

    
    review_button = driver.find_elements_by_id("reviews-medley-footer")[0]
    elems = review_button.find_elements_by_css_selector("a")
    for e in elems:
        link = e.get_attribute("href")
    driver.get(link)
    first_box = driver.find_element_by_id("cm_cr-rvw_summary-viewpoints")
    second_box_find = driver.find_elements_by_css_selector("div")
    for candidate in second_box_find:
        if "a-row view-point" == candidate.get_attribute("class"):
            second_box = candidate
    pos_crit_list = second_box.find_elements_by_css_selector("div")
    for pos_crit in pos_crit_list:
        if "view-point-review positive-review" in pos_crit.get_attribute("class"):
            pos_box = pos_crit
        if "view-point-review critical-review" in pos_crit.get_attribute("class"):
            crit_box = pos_crit

    pos_a_elements = pos_box.find_elements_by_css_selector("a")
    for a in pos_a_elements:
        print(a.text)
        if a.text == "All positive reviews":
            pos_link = a.get_attribute("href")

    crit_a_elements = crit_box.find_elements_by_css_selector("a")
    for a in crit_a_elements:
        print(a.text)
        if a.text == "All critical reviews":
            crit_link = a.get_attribute("href")



    pos_reviews_lod = get_reviews_lod(driver, pos_link, "pos")
    for review in pos_reviews_lod:
        print(review)
    crit_reviews_lod = get_reviews_lod(driver, crit_link, "crit")
    for review in crit_reviews_lod:
        print(review)

    reviews_lod = pos_reviews_lod + crit_reviews_lod

    print(len(pos_reviews_lod), len(crit_reviews_lod))
    
    with open("reviews_lod.json", "w") as outfile:
        json.dump(reviews_lod, outfile)


    driver.get(url)
    span_list = driver.find_elements_by_css_selector("span")
    for span in span_list:
        if span.get_attribute("data-widget-name") == "cr-summarization-attributes":
            find_span = span

    y = find_span.location['y']
    driver.execute_script("window.scrollTo(0, {});".format(y-100))

    elem = driver.find_element_by_id("reviewsMedley")
    span_list = elem.find_elements_by_css_selector("span")
    for span in span_list:
        if span.get_attribute("data-widget-name") == "cr-summarization-attributes":
            find_span = span

    
    
    divs = find_span.find_elements_by_css_selector("div")
    features = {}
    for div in divs:
        if "cr-summarization-attribute" in div.get_attribute("id"):
            found = False
            spans = div.find_elements_by_css_selector("span")
            for span in spans:
                if span.text.replace(" ", "").isalpha():
                    if span.text == "See more":
                        found = True
                        div.click()
                        see_more_div = div
                        break
                    feature = span.text
                    features[span.text] = 0
                elif span.text.replace(" ", "").replace(".", "").isnumeric():
                    features[feature] = float(span.text)
            if found:
                ahrefs = see_more_div.find_elements_by_css_selector("a")
                break
    
    try:
        for aherf in ahrefs:
            if "a-expander-toggle" in aherf.get_attribute("data-action"):
                aherf.click()
    
        expanded_div = see_more_div.find_element_by_id("cr-summarization-attributes-expanded")
        new_divs = expanded_div.find_elements_by_css_selector("div")
        for new_div in new_divs:
            if "cr-summarization-attribute" in new_div.get_attribute("id"):
                spans = new_div.find_elements_by_css_selector("span")
                for span in spans:
                    if span.text.replace(" ", "").isalpha():
                        feature = span.text
                        features[span.text] = 0
                    elif span.text.replace(" ", "").replace(".", "").isnumeric():
                        features[feature] = float(span.text)
    except NameError:
        pass
    
    

    f = open("reviews_lod.json", "r")
    reviews_lod = json.load(f)

    product_name = "Dyson Vacuum Cleaner"
    amazon_features = {'Maneuverability': 4.7, 'Easy to clean': 4.7, 'Suction power': 4.7, 'Light weight': 4.6, 'Battery life': 4.5}


    product = product_name.split()
    product_def = []

    doc = nlp(product_name)

    definition, word_dist = get_wiki_def(product_name)
    if not isinstance(definition, list):
        for ent in doc.ents:
            definition = get_wiki_def(ent.text)
            if isinstance(definition, list):
                product_name = ent.text
                break

    product_def, word_dist = definition
    chunks = chunker.parse(nltk.pos_tag(nltk.word_tokenize(definition[0])))
    willDef = False
    np = []
    for word, pos, chunktag in chunks:
        if chunktag == "B-NP" and willDef:
            np.append((word, pos))
        if chunktag == "I-NP" and willDef:
            np.append((word, pos))
        if chunktag == "O":
            if "VBZ" in pos or "VBP" in pos:
                willDef = True
            elif len(np) > 0:
                break

    product_type = ""
    for word, pos in np:
        if pos == "NN":
            product_type = word

    if len(product_type) == 0:
        print("what?")
        found = False
        for i in range(len(product)):
            for j in range(len(product), i, -1):
                sliced_name = " ".join(product[i:j])
                print(sliced_name)
                product_def, word_dist = get_wiki_def(sliced_name)
                if product_def != "blank":
                    chunks = chunker.parse(nltk.pos_tag(nltk.word_tokenize(product_def[0])))
                    willDef = False
                    np = []
                    for word, pos, chunktag in chunks:
                        if chunktag == "B-NP" and willDef:
                            np.append((word, pos))
                        if chunktag == "I-NP" and willDef:
                            np.append((word, pos))
                        if chunktag == "O":
                            if "VBZ" in pos or "VBP" in pos:
                                willDef = True
                            elif len(np) > 0:
                                break
                    for word, pos in np:
                        if pos == "NN":
                            product_type = word
                if len(product_type) > 0:
                    found = True
                    break
            if found:
                break


    print(amazon_features, product_def, product_type, product)
    """"
    
    review_ents = set()
    fdist = nltk.FreqDist()
    pos_fdist = nltk.FreqDist()
    crit_fdist = nltk.FreqDist()
    pos_tag_cfdist = nltk.ConditionalFreqDist()
    fdist_ents = nltk.FreqDist()

    for review in reviews_lod:
        words = nltk.word_tokenize(review["content"])
        pos_tag = nltk.pos_tag(words, tagset="universal")
        doc = nlp(review["content"])
        for word, pos in pos_tag:
            if word not in punct and word.lower() not in sw and word.isalpha():
                isProd = False
                for prod in product:
                    if word.lower() == prod.lower():
                        isProd = True
                        break
                if not isProd:
                    pos_tag_cfdist[word][pos] += 1
                    fdist[word] += 1
                    if review["polarity"] == "pos":
                        pos_fdist[word] += 1
                    else:
                        crit_fdist[word] += 1
        for ent in doc.ents:
            if ent.text.lower() not in product_name.lower():
                review_ents.add(ent.text)
                words = nltk.word_tokenize(ent.text)
                for word in words:
                    if word not in punct and word.lower() not in sw and word.isalpha():
                        fdist_ents[word] += 1

    fdist = fdist.most_common()
    fdist_ents = fdist_ents.most_common()
    pos_fdist = pos_fdist.most_common()
    crit_fdist = crit_fdist.most_common()
    
    with open("fdist.json", "w") as outfile:
        json.dump(fdist, outfile)

    with open("fdist_ents.json", "w") as outfile:
        json.dump(fdist_ents, outfile)

    with open("pos_fdist.json", "w") as outfile:
        json.dump(pos_fdist, outfile)

    with open("crit_fdist.json", "w") as outfile:
        json.dump(crit_fdist, outfile)

    with open("pos_tag_cfdist.json", "w") as outfile:
        json.dump(pos_tag_cfdist, outfile)
    
    ent_wiki_fdist = defaultdict(float)
    ent_defs = {}
    for ent, score in fdist_ents:
        ent_wiki, word_wiki = get_wiki_def(ent)
        if (isinstance(ent_wiki, list)) and len(ent_wiki) > 0:
            ent_defs[ent] = ent_wiki[0]
            vals = list(word_wiki.values())
            length = sum(vals)
            for word in word_wiki:
                if word not in ent and word.isalpha():
                    ent_wiki_fdist[word] += word_wiki[word] / length

    with open("ent_wiki_fdist.json", "w") as outfile:
        json.dump(ent_wiki_fdist, outfile)
        
    with open("ent_defs.json", "w") as outfile:
        json.dump(ent_defs, outfile)
    
    print(len(fdist))
    print(len(ent_wiki_fdist))
    
    f = open("fdist.json", "r")
    list_fdist = json.load(f)
    fdist = {}
    for word, freq in list_fdist:
        fdist[word] = freq
    f = open("ent_wiki_fdist.json", "r")
    ent_wiki_fdist = json.load(f)
    f = open("pos_tag_cfdist.json", "r")
    pos_tag_cfdist = json.load(f)

    token1 = sim("quality")[0]
    token2 = sim("feature")[0]
    token3 = sim(product_type)[0]
    token_product = sim("product")[0]
    slices = list(ent_wiki_fdist.keys())[:200]
    words = [w for w, s in list_fdist[:300]] + slices

    scores = []
    word_set = set()

    for word in words:
        max = 0
        max_pos = ""
        add = False

        try:
            for pos in pos_tag_cfdist[word.lower()]:
                if pos_tag_cfdist[word.lower()][pos] > max:
                    max = pos_tag_cfdist[word][pos]
                    max_pos = pos
                    if max_pos == "NOUN":
                        add = True
        except KeyError:
            if word in slices:
                add = True
        if not add:
            continue
        if word in product_name or word == product_type:
            add = False
        for def_word, def_pos in np:
            if word.lower() == def_word.lower():
                add = False
                break
        if add:
            token4 = sim(word)[0]
            sim1, sim2, sim3 = token1.similarity(token4), token2.similarity(token4), token3.similarity(token4)
            sim_product = token_product.similarity(token4)
            if sim2 > 0.35 and sim3 < 0.9 and word not in word_set and sim_product < 0.7:
                word_set.add(word)
                print(word, sim2)
                weighted = sim2 * 4 + sim3 * 6
                

                scores.append((word, weighted))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print(scores)
    num_feat = 0
    features = []
    for word, score in scores:
        will_add = True
        word_token = sim(word)[0]
        if num_feat == 5:
            break
        if stemmer.stem(word) in features:
            will_add = False
        for chosen in features:
            if word_token.similarity(sim(chosen)[0]) > 0.6:
                will_add = False
                break
        if not will_add:
            continue
        features.append(word)
        num_feat += 1

    print(features)


    
    features = ['system', 'tool', 'function', 'technology', 'use']
    f = open("fdist.json", "r")
    fdist = json.load(f)
    f = open("fdist_ents.json", "r")
    fdist_ents = json.load(f)
    f = open("pos_tag_cfdist.json", "r")
    pos_tag_cfdist = json.load(f)
    f = open("ent_defs.json", "r")
    ent_defs = json.load(f)


    ne_list = [w for w, score in fdist_ents if w.lower() not in product_name.lower() and w.lower() not in product_type]
    fdist_list = [w for w, score in fdist if w not in ne_list and w.lower() not in product_name.lower() and product_name.lower() not in w.lower()  and w.lower() not in product_name.lower() and product_type.lower() not in w.lower()]
    features.extend(['Noise level', 'Value for money', 'Tech Support', 'Sheerness'])
    feature_reviews = {}
    for feature in features:
        sims = []
        feature_tokens = sim(feature)
        for w in fdist_list:
            w_token = sim(w)[0]
            for feature_token in feature_tokens:
                if feature_token.similarity(w_token) > 0.7:
                    print(feature, w, feature_token.similarity(w_token))
                    if not w.lower() in feature.lower() and feature.lower() not in w.lower():
                        sims.append(w)
                    break
        features = [feature] + sims
        comwords = [w for w in feature.lower().split() if w not in sw]
        for w in ent_defs:
            for f in features:
                if f.lower() in ent_defs[w].lower():
                    comwords.extend(w.split())
        comwords.extend(product)
        comwords.extend([w for w, p in np if w.lower() not in sw])
        comwords.extend(features)
        comwords = list(set(comwords))
        keylist = toggleComments(features, comwords, reviews_lod)
        avg_score = keylist[0]
        avg_polarity_score = keylist[1]
        feature_reviews[feature] = keylist
        reviews = keylist[2:12]
        for rev in reviews:
            post_score, polarity_score, title, content, title_highlight_indices, content_highlight_indices, index = rev
            print(post_score, polarity_score)
            for i, word in enumerate(nltk.word_tokenize(title)):
                if i in title_highlight_indices:
                    print("\"\"{}\"\"".format(word),end=" ")
                else:
                    print(word, end=" ")
            print()
            for i, word in enumerate(nltk.word_tokenize(content)):
                if i in content_highlight_indices:
                    print("\"\"{}\"\"".format(word),end=" ")
                else:
                    print(word, end=" ")
            print()
        print()


    with open("feature_reviews.json", "w") as outfile:
        json.dump(feature_reviews, outfile)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
f  = open(json file)
feature_array_output = json.load(f)
"""
