
# coding: utf-8

# ## Please create an account on Plotly to successfully execute this Demo
# import plotly
# plotly.tools.set_credentials_file(username='<your_username>', api_key='<your_API_key>')

# In[1]:


import pandas as pd
input_df_file = "tokenizedDataframe.pickle"
filtereddata = pd.read_pickle(input_df_file)


# In[2]:


## Cleaning and processing after performing NLTK
def forSankey(dataframe, placeholder):
    import pandas as pd

    filtereddata = pd.read_pickle(input_df_file)

    ## Separate data frames for each product
    onlydata = filtereddata[filtereddata['product'] == placeholder.lower()]
    
    ## Renumbering the index 0 to length of dataframe
    onlydata.index = range(len(onlydata))
        
    ## Creating unique positive words
    pos_dict = []
    for i in range(len(onlydata)):
        if onlydata['SA'][i] == 1:
            for x in onlydata['words'][i]:
                pos_dict.append(x)
                
    # Creating a dictionary of frequencies of words
    pos_frequencies = []
    for w in pos_dict:
        pos_frequencies.append(pos_dict.count(w))
    pos_freq = {}
    for i, j in zip(pos_dict, pos_frequencies):
        pos_freq[i] = j

    # Removes elements from the list "pos_dict" if frequency is 3 or less than 3
    pos_words = pos_dict
    for i in pos_freq:
        if pos_freq[i] < 4:
            pos_words = [y for y in pos_words if y != i]
    pos_dict_unique = set(pos_words)
    pos_dict_unique = list(pos_dict_unique)

    ## Creating unique negative words
    
    neg_dict = []
    for i in range(len(onlydata)):
        if onlydata['SA'][i] == -1:
            for x in onlydata['words'][i]:
                neg_dict.append(x)
                
    # Creating a dictionary of frequencies of words
    neg_frequencies = []
    for w in neg_dict:
        neg_frequencies.append(neg_dict.count(w))
    neg_freq = {}
    for i, j in zip(neg_dict, neg_frequencies):
        neg_freq[i] = j

    # Removes elements from the list "neg_dict" if frequency is 3 or less than 3
    neg_words = neg_dict
    for i in neg_freq:
        if neg_freq[i] < 4:
            neg_words = [y for y in neg_words if y != i]
    neg_dict_unique = set(neg_words)
    neg_dict_unique = list(neg_dict_unique)

    ## Creating unique neutral words
    neu_dict = []
    for i in range(len(onlydata)):
        if onlydata['SA'][i] == 0:
            for x in onlydata['words'][i]:
                neu_dict.append(x)
                
    # Creating a dictionary of frequencies of words
    neu_frequencies = []
    for w in neu_dict:
        neu_frequencies.append(neu_dict.count(w))
    neu_freq = {}
    for i, j in zip(neu_dict, neu_frequencies):
        neu_freq[i] = j

    # Removes elements from the list "neg_dict" if frequency is 3 or less than 3
    neu_words = neu_dict
    for i in neu_freq:
        if neu_freq[i] < 4:
            neu_words = [y for y in neu_words if y != i]
    neu_dict_unique = set(neu_words)
    neu_dict_unique = list(neu_dict_unique)

    ## Sankey diagram label list for processing
    labellist = []
    labellist.append(placeholder)
    labellist.append("positive")
    labellist.append("negative")
    labellist.append("neutral")

    ## Calls the positivewords function which will give the unique list of positive words
    length_pos = len(pos_dict_unique)

    ## Appends all positive words to the list
    for i in pos_dict_unique:
        labellist.append(i)

    ## Calls the negativewords function which will give the unique list of negative words
    length_neg = len(neg_dict_unique)

    ## Appends all negative words to the list
    for i in range(length_neg):
        labellist.append(neg_dict_unique[i])

     ## Calls the neutralwords function which will give the unique list of neutral words
    length_neu = len(neu_dict_unique) 

     ## Appends all neutral words to the list
    for i in range(length_neu):
        labellist.append(neu_dict_unique[i])

    labellist = sorted(set(labellist), key=labellist.index)
    

    ## A sankey diagram is a visualization used to depict a flow from one set of values to another 
    ## The things being connected are called nodes and the connections are called links
    ## Sankeys are best used when you want to show a many-to-many mapping between two domains
    sourcelist = [0,0,0]

    for i in range(len(pos_dict_unique)):
        sourcelist.append(1)
    for i in range(len(neg_dict_unique)):
        sourcelist.append(2)
    for i in range(len(neu_dict_unique)):
        sourcelist.append(3)

    targetlist = [1,2,3]
    valuelist = [8,8,8]
    colorlist = ['rgb(4, 192, 201)',"rgb(155, 232, 29)","Red","Grey"]
    colorlinks = ["rgb(178,223,138)","rgb(248, 27, 62)","rgb(220, 220, 220)"]
    
    for i in pos_dict_unique:
        index = labellist.index(i)
        targetlist.append(index)
        valuelist.append(1)
        colorlist.append("rgb(4, 192, 201)")
        colorlinks.append("rgb(178,223,138)")
    
    for i in neg_dict_unique:
        index = labellist.index(i)
        targetlist.append(index)
        valuelist.append(1)
        colorlist.append("rgb(4, 192, 201)")
        colorlinks.append("rgb(248, 27, 62)")
    
    for i in neu_dict_unique:
        index = labellist.index(i)
        targetlist.append(index)
        valuelist.append(1)
        colorlist.append("rgb(4, 192, 201)")
        colorlinks.append("rgb(220, 220, 220)")

    import plotly
    from plotly.graph_objs import Scatter, Layout
    import plotly.plotly as py
    
    data = dict(
        type='sankey',
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(
            color = "black",
            width = 0.5
          ),
          label = labellist,
          color = colorlist
        ),
        link = dict(
          source = sourcelist,
          target = targetlist,
          value = valuelist,
          color = colorlinks

      ))

    layout =  dict(
        title = "Sankey Diagram: " + placeholder.title() ,
        font = dict(
          size = 10
        )
    )

    fig = dict(data=[data], layout=layout)
    p_obj = display_sankey(fig, placeholder)
    return p_obj


# In[ ]:


## Displaying Sankey diagrams
def display_sankey(fig, placeholder):
    import plotly
    from plotly.graph_objs import Scatter, Layout
    import plotly.plotly as py
    
    if placeholder == 'samsung':
        
        figure_kws = {'x_range': (0, 100), 'y_range': (100, 0), 'plot_height': 500, 'plot_width': 800}
    
        p1 = figure(**figure_kws)
        p1.axis.visible = False
        p1.background_fill_color = 'white'
        p1.border_fill_color = 'gray'
        p1.outline_line_color = None
        p1.grid.grid_line_color = None
        
        # The URL of the image saved in plotly account
        pl1 = py.plot(fig, filename = 'product-summary-samsung')
        pl_jpg1 = pl1 + '/sankey-diagram-samsung.jpg'
        p1.image_url(url=[pl_jpg1], x=[0], y=[0], w=[100], h=[100])
        p1.axis.visible = False
        p1.background_fill_color = 'white'
        p1.border_fill_color = '#74c476'
        p1.title.text_color = 'white'
        p1.toolbar.logo = None
        p1.toolbar_location = None
        return p1
    
    elif placeholder == 'apple':
        
        figure_kws = {'x_range': (0, 100), 'y_range': (100, 0), 'plot_height': 500, 'plot_width': 800}

        p2 = figure(**figure_kws)
        p2.axis.visible = False
        p2.background_fill_color = 'white'
        p2.border_fill_color = 'gray'
        p2.outline_line_color = None
        p2.grid.grid_line_color = None

        pl2 = py.plot(fig, filename = 'product-summary-apple')
        pl_jpg2 = pl2 + '/sankey-diagram-apple.jpg'
        p2.image_url(url=[pl_jpg2], x=[0], y=[0], w=[100], h=[100])
        p2.axis.visible = False
        p2.background_fill_color = 'white'
        p2.border_fill_color = '#74c476'
        p2.title.text_color = 'white'
        p2.toolbar.logo = None
        p2.toolbar_location = None
        return p2


# In[ ]:


## wordcloud for postive words
def poswordcloud(dataframe, polarity, placeholder):
    
    from wordcloud import WordCloud
    import pandas as pd
    from bokeh.plotting import figure 

    filtereddata = pd.read_pickle(dataframe)
   
    # Generate a word cloud image
    onlydata = filtereddata[filtereddata['product'] == placeholder.lower()]
    
    ## Renumbering the index 0 to length of dataframe
    onlydata.index = range(len(onlydata))
    pos_dict = []
    for i in range(len(onlydata)):
        if onlydata['SA'][i] == 1:
            for x in onlydata['words'][i]:
                pos_dict.append(x)

    wordcloud = WordCloud(background_color='white',collocations = False).generate(str(pos_dict))
    
    # Save the generated image to the local folder:
    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    imgName = polarity + '_word_cloud_' + placeholder + '.jpg'
    plt.savefig(imgName)
    
    p_obj = display_word_cloud(imgName, polarity, placeholder)
    return p_obj


# In[5]:


## wordcloud for negative words
def negwordcloud(dataframe, polarity, placeholder):
    
    from wordcloud import WordCloud
    import pandas as pd

    filtereddata = pd.read_pickle(dataframe)
   
    # Generate a word cloud image
    onlydata = filtereddata[filtereddata['product'] == placeholder.lower()]
    
    ## Renumbering the index 0 to length of dataframe
    onlydata.index = range(len(onlydata))
    neg_dict = []
    for i in range(len(onlydata)):
        if onlydata['SA'][i] == -1:
            for x in onlydata['words'][i]:
                neg_dict.append(x)

    wordcloud = WordCloud(background_color='white',collocations = False).generate(str(neg_dict))

    # Save the generated image to the local folder:
    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    imgName = polarity + '_word_cloud_' + placeholder + '.jpg'
    plt.savefig(imgName)
    
    p_obj = display_word_cloud(imgName, polarity, placeholder)
    return p_obj


# In[6]:


## wordcloud for neutral words
def neuwordcloud(dataframe, polarity, placeholder):
    
    from wordcloud import WordCloud
    import pandas as pd

    filtereddata = pd.read_pickle(dataframe)
   
    # Generate a word cloud image
    onlydata = filtereddata[filtereddata['product'] == placeholder.lower()]
    
    ## Renumbering the index 0 to length of dataframe
    onlydata.index = range(len(onlydata))
    neu_dict = []
    for i in range(len(onlydata)):
        if onlydata['SA'][i] == 0:
            for x in onlydata['words'][i]:
                neu_dict.append(x)

    wordcloud = WordCloud(background_color='white',collocations = False).generate(str(neu_dict))

    # Save the generated image to the local folder:
    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    imgName = polarity + '_word_cloud_' + placeholder + '.jpg'
    plt.savefig(imgName)
    
    p_obj = display_word_cloud(imgName, polarity, placeholder)
    return p_obj


# In[ ]:


############################################################################
# Displaying the word cloud image
def display_word_cloud(image_path, polarity, placeholder):

    # Open image, and make sure it's RGB*A*
    lena_img = Image.open(image_path).convert('RGBA')
    xdim, ydim = lena_img.size

    # Create an array representation for the image `img`, and an 8-bit "4
    # layer/RGBA" version of it `view`.
    img = np.empty((ydim, xdim), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))

    #print("view: ", view)
    # Copy the RGBA image into view, flipping it so it comes right-side up with a lower-left origin
    view[:,:,:] = np.flipud(np.asarray(lena_img))
    #print("view: ", view)

    # Display the 32-bit RGBA image
    dim = max(xdim, ydim)
    
    if placeholder == 'samsung':
        if polarity == 'Positive':
            name = "Samsung Positive Word Cloud"
            p1 = figure(title=name, x_range=(0,150), y_range=(0,75), plot_height=325, plot_width=400, toolbar_location = None)
            p1.image_rgba(image=[img], x=0, y=0, dw=150, dh=75)
            p1.axis.visible = False
            p1.background_fill_color = 'white'
            p1.border_fill_color = '#74c476'
            p1.title.text_color = 'white'
            p1.toolbar.logo = None
            p1.toolbar_location = None
            return p1
        elif polarity == 'Neutral':
            name = "Samsung Neutral Word Cloud"
            p1 = figure(title=name, x_range=(0,150), y_range=(0,75), plot_height=325, plot_width=400, toolbar_location = None)
            p1.image_rgba(image=[img], x=0, y=0, dw=150, dh=75)
            p1.axis.visible = False
            p1.background_fill_color = 'white'
            p1.border_fill_color = '#74c476'
            p1.title.text_color = 'white'
            p1.toolbar.logo = None
            p1.toolbar_location = None
            return p1
        elif polarity == 'Negative':
            name = "Samsung Negative Word Cloud"
            p1 = figure(title=name, x_range=(0,150), y_range=(0,75), plot_height=325, plot_width=400, toolbar_location = None)
            p1.image_rgba(image=[img], x=0, y=0, dw=150, dh=75)
            p1.axis.visible = False
            p1.background_fill_color = 'white'
            p1.border_fill_color = '#74c476'
            p1.title.text_color = 'white'
            p1.toolbar.logo = None
            p1.toolbar_location = None
            return p1        
    elif placeholder == 'apple':
        if polarity == 'Positive':
            name = "Apple Positive Word Cloud"
            p2 = figure(title=name, x_range=(0,150), y_range=(0,75), plot_height=325, plot_width=400, toolbar_location = None)
            p2.image_rgba(image=[img], x=0, y=0, dw=150, dh=75)
            p2.axis.visible = False
            p2.background_fill_color = 'white'
            p2.border_fill_color = '#74c476'
            p2.title.text_color = 'white'
            p2.toolbar.logo = None
            p2.toolbar_location = None
            return p2
        elif polarity == 'Neutral':
            name = "Apple Neutral Word Cloud"
            p2 = figure(title=name, x_range=(0,150), y_range=(0,75), plot_height=325, plot_width=400, toolbar_location = None)
            p2.image_rgba(image=[img], x=0, y=0, dw=150, dh=75)
            p2.axis.visible = False
            p2.background_fill_color = 'white'
            p2.border_fill_color = '#74c476'
            p2.title.text_color = 'white'
            p2.toolbar.logo = None
            p2.toolbar_location = None
            return p2
        elif polarity == 'Negative':
            name = "Apple Negative Word Cloud"
            p2 = figure(title=name, x_range=(0,150), y_range=(0,75), plot_height=325, plot_width=400, toolbar_location = None)
            p2.image_rgba(image=[img], x=0, y=0, dw=150, dh=75)
            p2.axis.visible = False
            p2.background_fill_color = 'white'
            p2.border_fill_color = '#74c476'
            p2.title.text_color = 'white'
            p2.toolbar.logo = None
            p2.toolbar_location = None
            return p2


# In[7]:


## Choropleth maps provide an easy way to visualize how a measurement varies across a geographic area
## It also shows the level of variability within a region
def choroplethdiagram(dataframe, placeholder):
    import pandas as pd
    filtereddata = pd.read_pickle(dataframe)
    onlydata = filtereddata[filtereddata['product'] == placeholder.lower()]
    ## Renumbering the index 0 to length of dataframe
    onlydata.index = range(len(onlydata))

    choromap = onlydata.copy()
    
    ## CSV file which holds all the country with respective country code
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
    countrydict = {}
    for i in range(len(df['COUNTRY'])):
        countrydict[df['COUNTRY'][i].lower()] = df['CODE'][i]
    choromap['place'] = None
    for key, value in countrydict.items():
            for i in range(len(choromap['location'])):
                if key in choromap['location'][i]:
                    choromap['place'][i] = countrydict[key]
    countrylist = {}

    countries = []
    countries_name_list = []
    for i in range(len(choromap['place'])):
        if choromap['place'][i] not in countries:
            countries.append(choromap['place'][i])
            countpos = 0
            countneg = 0
            countneu = 0
            for j in range(len(choromap['SA'])):
                if choromap['SA'][j] == 1:
                    if choromap['place'][j] == choromap['place'][i]:
                        countpos = countpos + 1
                if choromap['SA'][j] == -1:
                    if choromap['place'][j] == choromap['place'][i]:
                        countneg = countneg + 1
                if choromap['SA'][j] == 0:
                    if choromap['place'][j] == choromap['place'][i]:
                        countneu = countneu + 1

            countrylist[choromap['place'][i]] = [countpos,countneg,countneu]
            countries_name_list.append(choromap['location'][i])
    countrylist_new = {}
    
    ## The following portion helps us identitfy which attribute dominates a country ( either pos, neg or neu)
    for key,value in countrylist.items():
        list1 = []
        list1 = countrylist[key]
        if len(list1) == len(set(list1)):
            maximum = max(list1)
            countrylist_new[key] = list1.index(maximum)
        else:
            if len(set(list1)) == 1:
                countrylist_new[key] = 2
            else:
                list2 = []
                list3 = []
                list2 = set(list1)
                maximum2 = max(list2)
                index = 0
                for i in range(index,3):
                    if maximum2 == list1[i]:
                        indexofeach = i
                        list3.append(indexofeach)
                        index = indexofeach + 1
                if len(list3) > 1:
                    
                    ## If count remains the same for pos and neg, then assign the country to neutral
                    if ( 0 in list3) & ( 1 in list3) & ( 2 not in list3):
                        countrylist_new[key] =  2
                    
                    ## If count remains the same for pos and neu, then assign the country to positive
                    elif ( 0 in list3) & ( 2 in list3) & ( 1 not in list3):
                        countrylist_new[key] = 0
                        
                    ## If count remains the same for neg and neu, then assign the country to negative
                    elif ( 1 in list3) & ( 2 in list3) & ( 0 not in list3):
                        countrylist_new[key] = 1
                    
                    ## If count remains the same for pos, neg and neu, then assign the country to neutral
                    elif( 0 in list3) & ( 1 in list3) & ( 2 in list3):
                        countrylist_new[key] = 2
                else:
                    countrylist_new[key] = list3[0]


    mapdf = pd.DataFrame()
    countrymap = []
    countrycolor = []
    countryattribute = []
    count  = len(choromap['SA'])
    for key , value in countrylist_new.items():
        countrymap.append(key)
        countryattribute.append(value)
        if value is 0:
             countrycolor.append('rgb(178,223,138)')
        elif value is 1:
            countrycolor.append('rgb(248, 27, 62)')
        elif value is 2:
            countrycolor.append('rgb(220, 220, 220)')

    import plotly
    import pandas as pd
    import plotly.plotly as py
    
    data = [ dict(
            type = 'choropleth',
            locations = countrymap,
            z = countryattribute,
            text = countries_name_list,
            zmin = 0,
            zmax = 2,
            colorscale = [[0,'rgb(178,223,138)'],[0.5,'rgb(248, 27, 62)'],[1,'rgb(220, 220, 220)']],
            autocolorscale = False,
           visible = True,
           showscale = True,
            marker = dict(
                line = dict (
                    width = 1
                ) ),
           colorbar = dict(
                title = 'Sentiments',
                titleside = 'top',
                tickmode = 'array',
                tickvals = [0,1,2],
                ticktext = ['Positive','Negative','Neutral'],
                ticks = 'outside'
        ) )]

    layout = dict(
        title = 'Choropleth for ' + placeholder.title(),
        geo = dict(
            showframe = False,
            showcoastlines = False,
            projection = dict(
                type = 'Mercator'
            )
        )
    )

    fig = dict( data=data, layout=layout )

    p_obj = display_choropleth(fig, placeholder)
    return p_obj


# In[ ]:


## Displaying Sankey diagrams
def display_choropleth(fig, placeholder):
    import plotly
    from plotly.graph_objs import Scatter, Layout
    import plotly.plotly as py
    
    if placeholder == 'samsung':
        
        figure_kws = {'x_range': (0, 100), 'y_range': (100, 0), 'plot_height': 500, 'plot_width': 800}
    
        p1 = figure(**figure_kws)
        p1.axis.visible = False
        p1.background_fill_color = 'white'
        p1.border_fill_color = 'gray'
        p1.outline_line_color = None
        p1.grid.grid_line_color = None
        
        pl1 = py.plot(fig, filename = 'on-click-sankey')
        pl_jpg1 = pl1 + '/Choropleth-for-samsung.jpg'
        p1.image_url(url=[pl_jpg1], x=[0], y=[0], w=[100], h=[100])
        p1.axis.visible = False
        p1.background_fill_color = 'white'
        p1.border_fill_color = '#74c476'
        p1.title.text_color = 'white'
        p1.toolbar.logo = None
        p1.toolbar_location = None
        return p1
    
    elif placeholder == 'apple':
        
        figure_kws = {'x_range': (0, 100), 'y_range': (100, 0), 'plot_height': 500, 'plot_width': 800}

        p2 = figure(**figure_kws)
        p2.axis.visible = False
        p2.background_fill_color = 'white'
        p2.border_fill_color = 'gray'
        p2.outline_line_color = None
        p2.grid.grid_line_color = None

        pl2 = py.plot(fig, filename = 'on-click-sankey')
        pl_jpg2 = pl2 + '/Choropleth-for-apple.jpg'
        p2.image_url(url=[pl_jpg2], x=[0], y=[0], w=[100], h=[100])
        p2.axis.visible = False
        p2.background_fill_color = 'white'
        p2.border_fill_color = '#74c476'
        p2.title.text_color = 'white'
        p2.toolbar.logo = None
        p2.toolbar_location = None        
        return p2


# --------------------------------------

# ### User Interface with Bokeh

# In[5]:


from random import random

from bokeh.layouts import column
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc

from bokeh.layouts import gridplot, layout
from bokeh.palettes import Viridis3
from bokeh.plotting import figure
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox, column, Spacer
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import TextInput, Button, RadioButtonGroup #Dropdown # on_click
from bokeh.models.widgets import Panel, Tabs # title, # 
from bokeh.plotting import figure

from bokeh.models.widgets import Panel, Tabs
from bokeh.io import output_file, show
from bokeh.plotting import figure

import ipywidgets as wg
from IPython.display import display
from PIL import Image
from bokeh.models import Range1d
from bokeh.models.widgets import Div


# In[6]:


###################################################################################
## Sankey Plot
def sankey_analysis_handler():
    sankey_status.text = " "
    wordcloud_status.text = " "
    region_status.text = " "

    p1 = forSankey(input_df_file,"samsung")
    grid1 = gridplot([inputs, p1], ncols=2, width=1200)
    curdoc().clear()
    curdoc().add_root(project_title)
    curdoc().add_root(grid1)
    curdoc().title = "Market Value Product Summary"
    
    from time import sleep
    sleep(7)
    
    p2 = forSankey(input_df_file,"apple")
    grid2 = gridplot([blank_inputs1, p2], ncols=2, width=1200)
    curdoc().add_root(grid2)
    curdoc().title = "Market Value Product Summary"
    sankey_status.text = "Ran Sankey Analysis"
    
###################################################################################
## Word Cloud
def wordcloud_handler():
    sankey_status.text = " "
    wordcloud_status.text = " "
    region_status.text = " "

    p1 = poswordcloud(input_df_file, 'Positive', 'samsung')
    p2 = poswordcloud(input_df_file, 'Positive', 'apple')
    
    grid1 = gridplot([inputs, p1, p2], ncols=3, width=1200)
    curdoc().clear()
    curdoc().add_root(project_title)
    curdoc().add_root(grid1)
    curdoc().title = "Market Value Product Summary"
    #################################################################
    
    p3 = neuwordcloud(input_df_file, 'Neutral', 'samsung')
    p4 = neuwordcloud(input_df_file, 'Neutral', 'apple')
    
    grid2 = gridplot([blank_inputs1, p3, p4], ncols=3, width=1200)
    curdoc().add_root(grid2)
    #################################################################

    p5 = negwordcloud(input_df_file, 'Negative', 'samsung')
    p6 = negwordcloud(input_df_file, 'Negative', 'apple')
    
    grid3 = gridplot([blank_inputs2, p5, p6], ncols=3, width=1200)
    curdoc().add_root(grid3)

    wordcloud_status.text = "Ran WordCloud Analysis"


##################################################################################
## Chloropleth
def region_analysis_handler():
    sankey_status.text = " "
    wordcloud_status.text = " "
    region_status.text = " "

    p1 = choroplethdiagram(input_df_file,"samsung")
        
    grid1 = gridplot([inputs, p1], ncols=2, width=1200)
    curdoc().clear()
    curdoc().add_root(project_title)
    curdoc().add_root(grid1)
    curdoc().title = "Market Value Product Summary"
    
    from time import sleep
    sleep(7)
    
    p2 = choroplethdiagram(input_df_file,"apple")

    grid2 = gridplot([blank_inputs1, p2], ncols=2, width=1200)
    curdoc().add_root(grid2)
    curdoc().title = "Market Value Product Summary"

    #region_button.label = "Ran Region Analysis"
    region_status.text = "Ran Region Analysis"


# In[ ]:


project_title = Div(text="<b>Market Value Product Summary using Sentiment Analysis</b>", height=20, style={'font-size': '200%', 'color': '#393b79'})

###################################################################################
## Text Inputs
text_handler1 = TextInput(value="Apple", title="Product 1:")#, width=100)
text_handler2 = TextInput(value="Samsung", title="Product 2:")#, width=100)
spacing = Div(text="""  """, height=10) #,width=100)

###################################################################################
## Sankey
sankey_button = Button(label="Sankey Analysis", button_type="success") #, height=30)#, width=100)#, height=20)
sankey_button.on_click(sankey_analysis_handler)
sankey_status = Div(text="""  """, height=10) #,width=100)

###################################################################################
## Word Cloud
wordcloud_button = Button(label="Word Cloud", button_type="success")
wordcloud_button.on_click(wordcloud_handler)
wordcloud_status = Div(text="""  """, height=10) #,width=100)

###################################################################################
## Choropleth
region_button = Button(label="Region Analysis", button_type="success")
region_button.on_click(region_analysis_handler)
region_status = Div(text="""  """, height=10) #,width=100)

###################################################################################
# Initializing Grid contents
inputs = widgetbox([text_handler1, text_handler2, spacing,
                    sankey_button, sankey_status, 
                    wordcloud_button, wordcloud_status, 
                    region_button, region_status], height=700, width=300) #, tabs)# offset, amplitude, phase, freq)

blank_inputs1 = widgetbox([], height=700, width=300) #, tabs)# offset, amplitude, phase, freq)
blank_inputs2 = widgetbox([], height=700, width=300) #, tabs)# offset, amplitude, phase, freq)


# In[9]:


grid = gridplot([inputs], ncols=2, width=1200)#, toolbar_options=dict(logo=None)

curdoc().add_root(project_title)
curdoc().add_root(grid)
curdoc().title = "Market Value Product Summary"


# ---------------------------------------------

# cd ".\<Your_Directory_Path>\Market Value Product Analysis\Project_Demo\"
# 
# bokeh serve Market_Value_Product_Summary_Demo.py
