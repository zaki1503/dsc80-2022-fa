# lab.py


import os
import pandas as pd
import numpy as np
import requests
import bs4
import lxml


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.
    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!
    >>> question1()
    >>> os.path.exists('lab06_1.html')
    True
    """
    # Don't change this function body!
    # No python required; create the HTML file.
    import textwrap
    file_html = textwrap.dedent(
        """
        <html>
            <head>
                <title> Untitled </title>
                </head>
                <body>
                    <h1> Primero Header for images
                    </h1>
                    <img src="./data/image.jpg">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/4/49/A_black_image.jpg">
                    <img src="missing" alt="lost to the sands of time">

                    <h1> Header para Search Engines </h1>
                    <a href= "https://www.google.com"> the verb,  </a>
                    <a href= "https://www.bing.com"> the challenger,  </a>
                    <a href= "https://duckduckgo.com"> the paranoid  </a>

                    <h1> Table </h1>
                    <table>
                        <tr>
                            <td> . </td>
                            <td>Blue</td>
                            <td>Brown</td>
                        </tr>
                        <tr>
                            <td> Blue </td>
                            <td> BB </td>
                            <td> BBr </td>
                        </tr>
                        <tr>
                            <td> Brown </td>
                            <td> BBr </td>
                            <td> BrBr </td>
                        </tr>
                    </table>
                </body>
            </html>
        """
    )

    f = open("lab06_1.html", "w")
    f.write(file_html)
    f.close()
    
    return


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------
def helper(page):
    price = page.find('p', attrs={'class': 'price_color'}).text
    price = float(''.join([x for x in price if (x.isdigit() or x == '.')]))

    def has_star_rating(x):
        return 'star-rating' in x if x else False

    rating = (
        page
        .find('p', attrs={'class': lambda })
        .get('class')[-1]
    )

    str2int = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
    if (str2int[rating] >= 4) and (price < 50):
        return True
    else:
        return False



def extract_book_links(text):
    """
    :Example:
    >>> fp = os.path.join('data', 'products.html')
    >>> out = extract_book_links(open(fp, encoding='utf-8').read())
    >>> url = 'scarlet-the-lunar-chronicles-2_218/index.html'
    >>> out[1] == url
    True
    """
    ...


def get_product_info(text, categories):
    """
    :Example:
    >>> fp = os.path.join('data', 'Frankenstein.html')
    >>> out = get_product_info(open(fp, encoding='utf-8').read(), ['Default'])
    >>> isinstance(out, dict)
    True
    >>> 'Category' in out.keys()
    True
    >>> out['Rating']
    'Two'
    """
    ...


def scrape_books(k, categories):
    """
    :param k: number of book-listing pages to scrape.
    :returns: a dataframe of information on (certain) books
    on the k pages (as described in the question).
    :Example:
    >>> out = scrape_books(1, ['Mystery'])
    >>> out.shape
    (1, 11)
    >>> out['Rating'][0] == 'Four'
    True
    >>> out['Title'][0] == 'Sharp Objects'
    True
    """
    ...


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def stock_history(ticker, year, month):
    """
    Given a stock code and month, return the stock price details for that month
    as a DataFrame.

    >>> history = stock_history('BYND', 2019, 6)
    >>> history.shape == (20, 13)
    True
    >>> history.label.iloc[-1]
    'June 03, 19'
    """
    ...


def stock_stats(history):
    """
    Given a stock's trade history, return the percent change and transactions
    in billions of dollars.

    >>> history = stock_history('BYND', 2019, 6)
    >>> stats = stock_stats(history)
    >>> len(stats[0]), len(stats[1])
    (7, 6)
    >>> float(stats[0][1:-1]) > 30
    True
    >>> float(stats[1][:-1]) > 1
    True
    >>> stats[1][-1] == 'B'
    True
    """
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def get_comments(storyid):
    """
    Returns a DataFrame of all the comments below a news story
    >>> out = get_comments(18344932)
    >>> out.shape
    (18, 5)
    >>> out.loc[5, 'by']
    'RobAtticus'
    >>> out.loc[5, 'time'].day
    31
    """
    ...
