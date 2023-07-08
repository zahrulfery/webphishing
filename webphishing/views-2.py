import imp
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import numpy as np
import datetime
import os
import joblib
import pickle
import signal
import re
import requests
import urllib.parse
from urllib.parse import urlparse
import tldextract
from datetime import datetime
from bs4 import BeautifulSoup
import whois
import time
import sys
import warnings
from pathlib import Path
import json

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical, plot_model


from .scripts import extract_url as eu
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
            
            print("extracted_domain : ", extracted_domain)
            print("tmp : ", tmp)
            print("pth : ", pth)
            print("path : ", path)
            print("tld : ", tld)
            print("parsed : ", parsed)
            print("scheme : ", scheme)
            Href, Link, Anchor, Media, Form, CSS, Favicon, IFrame, Title, Text = eu.extract_data_from_URL(
                hostname, content, domain, Href, Link, Anchor, Media, Form, CSS, Favicon, IFrame, Title, Text)
        else:
            return render(request, '404notfound.html', context={"url": urlweb})

        # domain_in_title = cf.domain_in_title(extracted_domain.domain, Title)
        # ip = uf.having_ip_address(url)
        # nb_eq = uf.count_equal(url)
        # prefix_suffix = uf.prefix_suffix(url)
        # empty_title = cf.empty_title(Title)
        # tld_in_subdomain = uf.tld_in_subdomain(tld, subdomain)
        # ratio_intMedia = cf.internal_media(Media)

        # data_input = [google_index, page_rank, nb_www, ratio_digits_url, domain_in_title, nb_hyperlinks, phish_hints, domain_age, ip, nb_qm, ratio_intHyperlinks, length_url, nb_slash, length_hostname, nb_eq, shortest_word_host, longest_word_path, ratio_digits_host, prefix_suffix, nb_dots, empty_title, longest_words_raw, tld_in_subdomain, length_words_raw, ratio_intMedia]
        # ini harusnya 35
        # data_input = [google_index, page_rank, nb_www, ratio_digits_url, domain_in_title, domain_age, nb_hyperlinks, phish_hints, ip, nb_qm, length_url, ratio_intHyperlinks, nb_slash, nb_eq,
        #               length_hostname, shortest_word_host, ratio_digits_host, empty_title, prefix_suffix, nb_dots, longest_word_path, avg_word_path, avg_word_host, tld_in_subdomain, longest_words_raw]
        
        print("url --- atas: ", url)
        print("hostname : ", hostname)
        print("domain : ", domain)
        
        print('---------------------------------------------------------------------------')
        # url = eu.is_URL_accessible(url)
        url_exists = eu.url_exists(url)
        print("url_exists : ", url_exists)
        length_url = uf.url_length(url)
        print("length_url : ", length_url)
        length_hostname = uf.url_length(hostname)
        print("length_hostname : ", length_hostname)
        nb_dots = uf.count_dots(url)
        print("nb_dots : ", nb_dots)
        nb_hyphens = uf.nb_hyphens(url)
        print("nb_hyphens : ", nb_hyphens)
        nb_qm = uf.count_exclamation(url)
        print("nb_qm : ", nb_qm)
        nb_slash = uf.count_slash(url)
        print("nb_slash : ", nb_slash)
        nb_space = uf.nb_space(url)
        print("nb_space : ", nb_space)
        nb_www = uf.check_www(words_raw)
        print("nb_www : ", nb_www)
        ratio_digits_url = uf.ratio_digits(url)
        print("ratio_digits_url : ", ratio_digits_url)
        ratio_digits_host = uf.ratio_digits(hostname)
        print("ratio_digits_host : ", ratio_digits_host)
        length_words_raw = uf.length_word_raw(words_raw)
        print("length_words_raw : ", length_words_raw)
        shortest_words_raw = uf.shortest_words_raw(words_raw)
        print("shortest_words_raw : ", shortest_words_raw)
        shortest_word_host = uf.shortest_word_length(words_raw_host)
        print("shortest_word_host : ", shortest_word_host)
        # print("--subdomain : ", subdomain)
        # print("--tld : ", tld)
        # print("--words_raw : ", words_raw)
        shortest_word_path = uf.shortest_word_path(subdomain,tld,words_raw)
        print("shortest_word_path : ", shortest_word_path)
        longest_words_raw = uf.longest_word_length(words_raw)
        print("longest_words_raw : ", longest_words_raw)

        host = (scheme, subdomain, domain)
        longest_word_host = uf.longest_word_host(host)
        print("longest_word_host : ", longest_word_host)
        longest_word_path = uf.longest_word_length(words_raw_path)
        print("longest_word_path : ", longest_word_path)
        avg_words_raw = uf.avg_words_raw(words_raw)
        print("avg_words_raw : ", avg_words_raw)
        avg_word_path = uf.average_word_length(words_raw_path)
        print("avg_word_path : ", avg_word_path)
        avg_word_host = uf.average_word_length(words_raw_host)
        print("avg_word_host : ", avg_word_host)
        phish_hints = uf.phish_hints(url)
        print("phish_hints : ", phish_hints)
        domain_in_brand = uf.domain_in_brand(url)
        print("domain_in_brand : ", domain_in_brand)
        nb_hyperlinks = cf.nb_hyperlinks(Href, Link, Media, Form, CSS, Favicon)
        print("nb_hyperlinks : ", nb_hyperlinks)
        ratio_intHyperlinks = cf.internal_hyperlinks(
            Href, Link, Media, Form, CSS, Favicon)
        print("ratio_intHyperlinks : ", ratio_intHyperlinks)
        ratio_extHyperlinks = cf.ratio_extHyperlinks(Href, Link, Media, Form, CSS, Favicon)
        print("ratio_extHyperlinks : ", ratio_extHyperlinks)
        nb_extCSS = cf.nb_extCSS(CSS)
        print("nb_extCSS : ", nb_extCSS)
        ratio_extRedirection = cf.ratio_extRedirection(Href, Link, Media, Form, CSS, Favicon)
        print("ratio_extRedirection : ", ratio_extRedirection)
        ratio_extErrors = cf.ratio_extErrors(Href, Link, Media, Form, CSS, Favicon)      
        print("ratio_extErrors : ", ratio_extErrors)
        links_in_tags = cf.links_in_tags(url)
        print("links_in_tags : ", links_in_tags)
        ratio_extMedia = cf.ratio_extMedia(Media)
        print("ratio_extMedia : ", ratio_extMedia)
        anchor_links = cf.get_anchor_links_from_url(url)
        # print("---anchor_links : ", anchor_links)
        safe_anchor = cf.safe_anchor(anchor_links)
        print("safe_anchor : ", safe_anchor)
        domain_registration_length = ef.domain_registration_length(domain)
        print("domain_registration_length : ", domain_registration_length)
        domain_age = ef.count_domain_age(domain)
        print("domain_age : ", domain_age)
        web_traffic = ef.web_traffic(url)
        print("web_traffic : ", web_traffic)
        google_index = ef.google_index(url)
        print("google_index : ", google_index)
        page_rank = ef.page_rank(api_key, domain)
        print("page_rank : ", page_rank)
        
        data_input = [url_exists, length_url, length_hostname, nb_dots, nb_hyphens, nb_qm, nb_slash, nb_space, nb_www, ratio_digits_url, ratio_digits_host, length_words_raw, shortest_words_raw, shortest_word_host, shortest_word_path, longest_words_raw, longest_word_host, longest_word_path, avg_words_raw, avg_word_host, avg_word_path, phish_hints, domain_in_brand, nb_hyperlinks, ratio_intHyperlinks, ratio_extHyperlinks, nb_extCSS, ratio_extRedirection, ratio_extErrors, links_in_tags, ratio_extMedia, safe_anchor, domain_registration_length, domain_age, web_traffic, google_index, page_rank]
        print("data_input : ", data_input)
        
        count_array = len(data_input)
        print("Count of elements in the array:", count_array)

        path = str(Path(__file__).resolve().parent.parent)
        # scaler_load = joblib.load(path + '/webphishing/scripts/std_scaler.bin')
        # scaler_load = joblib.load(
        #     path + '/Tugas Akhir/Pengujian/Test/model-fs-25/std_scaler.bin')
        # print('scaler_load:: ',repr(scaler_load))
        
        # # Display the attributes and methods of scaler_load
        # print(dir(scaler_load))

        
        # # Access the scaling factors for the features
        # scaling_factors = scaler_load.scale_

        # # Print the scaling factors for each feature
        # for feature_index, scaling_factor in enumerate(scaling_factors):
        #     print(f"Feature {feature_index + 1}: {scaling_factor}")
        # Filter out non-numeric elements and convert to float
        # numerical_data = [float(x) for x in data_input[1:] if isinstance(x, (int, float))]
        # numerical_data = np.array(numerical_data).reshape(1, -1)
        print('---------------------------------------------------------------------------')
        # load_model = tf.keras.models.load_model(
        #     path + '/Tugas Akhir/Pengujian/Test/model-fs-25/Model_Testing DecisionTree.h5')
        # load_model.summary()
        # # load_model = tf.keras.models.load_model(
        # #     path + '/webphishing/scripts/mlp_model_fs_25.h5')
        # # data_scale = scaler_load.transform(numerical_data)
        # # data_scale = scaler_load.transform([data_input])
        # load_model.compile(loss=[tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.MeanSquaredError()],
        #                    optimizer=tf.keras.optimizers.Adam(
        #                        learning_rate=0.0001),
        #                    metrics=['accuracy', tf.keras.metrics.MeanSquaredError()])
        # preds = load_model.predict([data_input])

        # data_input = np.array([1, 25, 16, 3, 0, 0, 3, 0, 1, 0.0, 0.0, 2, 3, 3, -1, 6, 0, 0, 4.5, 4.5, 0, 0, 0, 64, 0.765625, 0.234375, 1, 0.13333333333333333, 0.13333333333333333, 55.18394648829431, 0, 100.0, -1, -1, 0, -1, 5])
        # data_input = np.reshape(data_input, (1, 36))

        load_model = tf.keras.models.load_model(path + '/Tugas Akhir/Pengujian/Test/model-fs-25/Model_Testing DecisionTree.h5')
        # input_shape = load_model.layers[0].input_shape
        # print("Input shape:", input_shape)

        # Assuming data_input is a list or array with shape (37,)
        data_input = np.array(data_input)
        print('data_input[:-1]', data_input[:-1])
        print('data_input[:-1].reshape(1, -1)', data_input[:-1].reshape(1, -1))
        reshaped_input = data_input[:-1].reshape(1, -1)  # Remove the last element and reshape to (1, 36)


        load_model.compile(loss=[tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.MeanSquaredError()],
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                        metrics=['accuracy', tf.keras.metrics.MeanSquaredError()])
        # preds = load_model.predict(data_input)
        preds = load_model.predict(reshaped_input)


        result = np.argmax(preds, axis=1)
        print(preds[0])

        # return JsonResponse({"data_input": data_input, "data_scale": json.dumps(data_scale[0], cls=NpEncoder), "result": json.dumps(result[0], cls=NpEncoder)})

        return render(request, 'result.html', context={"url": url, "data_input": data_input, "data_scale": reshaped_input[0], "result": result, "probability": {
            "phishing": preds[0][0],
            "legitimate": preds[0][1]
        }})


def notfound(request):
    return render(request, '404notfound.html')
