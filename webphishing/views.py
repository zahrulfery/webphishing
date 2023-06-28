from django.shortcuts import render
import numpy as np
import os
import joblib
from urllib.parse import urlparse
import tldextract
from pathlib import Path
import json

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from .scripts import extract_url as eu  # Import the extract_url module
from .scripts import content_features as cf
from .scripts import external_features as ef
from .scripts import url_features as uf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def index(request):
    return render(request, 'home.html')


def result(request):
    if request.method == 'POST':
        urlweb = request.POST["url"]
        api_key = "c0sc88cccogs8g4c0c8osgowskg44ogs8ow4wk8w"  # One Page Rank API
        state, url, page = eu.is_URL_accessible(urlweb)
        Href = {'internals': [], 'externals': [], 'null': []}
        Link = {'internals': [], 'externals': [], 'null': []}
        Anchor = {'safe': [], 'unsafe': [], 'null': []}
        Media = {'internals': [], 'externals': [], 'null': []}
        Form = {'internals': [], 'externals': [], 'null': []}
        CSS = {'internals': [], 'externals': [], 'null': []}
        Favicon = {'internals': [], 'externals': [], 'null': []}
        IFrame = {'visible': [], 'invisible': [], 'null': []}
        Title = ''
        Text = ''

        if state:
            content = page.content
            hostname, domain, path = eu.get_domain(url)
            extracted_domain = tldextract.extract(url)
            domain = extracted_domain.domain + '.' + extracted_domain.suffix
            subdomain = extracted_domain.subdomain
            tmp = url[url.find(extracted_domain.suffix):len(url)]
            pth = tmp.partition("/")
            path = pth[1] + pth[2]
            words_raw, words_raw_host, words_raw_path = eu.words_raw_extraction(
                extracted_domain.domain, subdomain, pth[2])
            tld = extracted_domain.suffix
            parsed = urlparse(url)
            scheme = parsed.scheme
            Href, Link, Anchor, Media, Form, CSS, Favicon, IFrame, Title, Text = eu.extract_data_from_URL(
                hostname, content, domain, Href, Link, Anchor, Media, Form, CSS, Favicon, IFrame, Title, Text)
        else:
            return render(request, '404notfound.html', context={"url": urlweb})


        google_index = ef.google_index(url)
        page_rank = ef.page_rank(api_key, domain)
        nb_www = uf.check_www(words_raw)
        ratio_digits_url = uf.ratio_digits(url)
        domain_in_title = cf.domain_in_title(extracted_domain.domain, Title)
        nb_hyperlinks = cf.nb_hyperlinks(Href, Link, Media, Form, CSS, Favicon)
        phish_hints = uf.phish_hints(url)
        domain_age = ef.count_domain_age(domain)
        ip = uf.having_ip_address(url)
        nb_qm = uf.count_exclamation(url)
        ratio_intHyperlinks = cf.internal_hyperlinks(
            Href, Link, Media, Form, CSS, Favicon)
        length_url = uf.url_length(url)
        nb_slash = uf.count_slash(url)
        length_hostname = uf.url_length(hostname)
        nb_eq = uf.count_equal(url)
        shortest_word_host = uf.shortest_word_length(words_raw_host)
        longest_word_path = uf.longest_word_length(words_raw_path)
        ratio_digits_host = uf.ratio_digits(hostname)
        prefix_suffix = uf.prefix_suffix(url)
        nb_dots = uf.count_dots(url)
        empty_title = cf.empty_title(Title)
        longest_words_raw = uf.longest_word_length(words_raw)
        tld_in_subdomain = uf.tld_in_subdomain(tld, subdomain)
        length_words_raw = uf.length_word_raw(words_raw)
        ratio_intMedia = cf.internal_media(Media)
        avg_word_path = uf.average_word_length(words_raw_path)
        avg_word_host = uf.average_word_length(words_raw_host)

        data_input = [google_index, page_rank, nb_www, ratio_digits_url, domain_in_title, domain_age, nb_hyperlinks, phish_hints, ip, nb_qm, length_url, ratio_intHyperlinks, nb_slash, nb_eq,
                      length_hostname, shortest_word_host, ratio_digits_host, empty_title, prefix_suffix, nb_dots, longest_word_path, avg_word_path, avg_word_host, tld_in_subdomain, longest_words_raw]

        path = str(Path(__file__).resolve().parent.parent)

        scaler_load = joblib.load(
            path + '/Tugas Akhir/Pengujian/Test/model-fs-25/std_scaler.bin')
     
        # Define a function to adjust the feature set of data_input
        def adjust_features(data):
            # Adjust the features of data to match the scaler's 87 features
            # Modify this function according to the specific requirements of your data

            # Pad data with zeros to match 87 features
            adjusted_data = np.concatenate((data, np.zeros(87 - len(data))))

            return adjusted_data

        # Adjust the features of data_input
        adjusted_data = adjust_features(data_input)

        # Transform the adjusted_data using the loaded scaler
        data_scale = scaler_load.transform([adjusted_data])

        # Modify your model architecture to match the number of features
        # Update the input shape in the first layer of your model to (None, num_features)
        load_model = tf.keras.models.load_model(
            path + '/Tugas Akhir/Pengujian/Test/model-fs-25/Model_Testing DecisionTree.h5')

        # Create a new Sequential model
        new_model = tf.keras.Sequential()
        print('load_model:: ',load_model)
        new_model.add(load_model.layers[2])  # Assuming the next layer is the layer you want to add

        new_model.compile(loss=[tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.MeanSquaredError()],
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                        metrics=['accuracy', tf.keras.metrics.MeanSquaredError()])

        preds = new_model.predict(data_scale)
        result = np.argmax(preds, axis=1)
        print('result',result)
        # print(preds[0])

        return render(request, 'result.html', context={"url": url, "data_input": data_input, "data_scale": data_scale[0], "result": result, "probability": {
            "phishing": preds[0][0],
            "legitimate": preds[0][1]
        }})


def notfound(request):
    return render(request, '404notfound.html')
