import pandas as pd
import plotly.express as px  # (version 4.7.0)
import numpy as np
import ast
import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import plotly.graph_objects as go
import networkx as nx

#### NOTE: the current version of the code is still very messy; a cleaner version will follow in the future

app = dash.Dash(__name__)

# ------------------------------------------------------------------------------
# Import and clean data (importing csv into pandas)
df = pd.read_csv('tweets_210308.csv', sep=';', low_memory=False)

df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

df['count'] =list(np.repeat(1, len(df)))
df_days = df.groupby(
    [df['created_at'].dt.date, 'geschlecht']
                    ).mean()

df_days_counts = df.groupby([df['created_at'].dt.date, 'geschlecht']).count()
df_days['abs_count'] = df_days_counts['count']

df_days.reset_index(inplace=True)
#### party analyses
df_party = df.groupby(
    [df['geschlecht'], 'partei']
                    ).mean()

df_party_counts = df.groupby([df['geschlecht'], 'partei']).count()
df_party['abs_count'] = df_party_counts['count']
df_party.reset_index(inplace=True)

### individual analyses
df_person = df.groupby([df['name'], 'geschlecht', 'partei','user_handle']).mean()
df_person_counts = df.groupby([df['name'], 'geschlecht','partei','user_handle']).count()
df_person['abs_count'] = df_person_counts['count']
df_person.reset_index(inplace=True)

#extract @mentions (here falsely labelled as retweets
retweets = []
for i in df['addressees']:
    for j in ast.literal_eval(i):
        retweets.append(j)

retweet_counts = []
for i in df_person['user_handle']:
    retweet_counts.append(retweets.count(i))

df_person['retweet_count'] = retweet_counts
df_person  = df_person.sort_values(by = 'abs_count', ascending=False)

######## text stuff #######
#create stop word list and clean corpus
stop_words = stopwords.words('german')
stop_words.extend(stopwords.words('french'))
stop_words.extend(stopwords.words('italian'))
stop_words.extend(stopwords.words('english'))
stop_words.extend(['https', 'dass', 'rt', 'mehr', 'ab',
                   'co','schon', 'seit', 'gibt',
                   'amp', 'geht', 'wurde','jahren','wäre',
                   'beim', 'wer','viele','einfach', 'kommt',
                   'neue',
                   'hui','aujourd','sans','tous','cette','comme',
                   'les', 'plus','fait','très', 'faut',
                   '12', 'sempre','26','17']) #add useless stuff to stopwords

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
##document term matrices for de/fr/it/en
df_de = df[df['language']=='de']
df_pol_de = df_de.groupby('name').agg(lambda x: ''.join(set(x))).reset_index()
docs = list(df_pol_de['text'])
vec = CountVectorizer()
X = vec.fit_transform(docs)
dtm_de = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

df_fr = df[df['language']=='fr']
df_pol_fr = df_fr.groupby('name').agg(lambda x: ''.join(set(x))).reset_index()
docs = list(df_pol_fr['text'])
vec = CountVectorizer()
X = vec.fit_transform(docs)
dtm_fr = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

df_it = df[df['language']=='it']
df_pol_it = df_it.groupby('name').agg(lambda x: ''.join(set(x))).reset_index()
docs = list(df_pol_it['text'])
vec = CountVectorizer()
X = vec.fit_transform(docs)
dtm_it = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

df_en = df[df['language']=='en']
df_pol_en = df_en.groupby('name').agg(lambda x: ''.join(set(x))).reset_index()
docs = list(df_pol_en['text'])
vec = CountVectorizer()
X = vec.fit_transform(docs)
dtm_en = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

## top words
#de
words_women = get_top_n_words(df_pol_de[df_pol_de['geschlecht']=='frau']['text'], 50)
words_men = get_top_n_words(df_pol_de[df_pol_de['geschlecht']=='mann']['text'], 50)
word =[]
count = []

for i in words_women:
    word.append(i[0])
    count.append(i[1])
for i in words_men:
    word.append(i[0])
    count.append(i[1])
words_top_n = pd.DataFrame({'word': word, 'count':count})
words_top_n['geschlecht'] = list(np.repeat(['frau','mann'], len(words_top_n)/2))

#fr
words_women = get_top_n_words(df_pol_fr[df_pol_fr['geschlecht']=='frau']['text'], 50)
words_men = get_top_n_words(df_pol_fr[df_pol_fr['geschlecht']=='mann']['text'], 50)

word =[]
count = []

for i in words_women:
    word.append(i[0])
    count.append(i[1])
for i in words_men:
    word.append(i[0])
    count.append(i[1])

words_top_fr = pd.DataFrame({'word': word, 'count':count})
words_top_fr['geschlecht'] = list(np.repeat(['frau','mann'], len(words_top_n)/2))

#it
words_women = get_top_n_words(df_pol_it[df_pol_it['geschlecht']=='frau']['text'], 50)
words_men = get_top_n_words(df_pol_it[df_pol_it['geschlecht']=='mann']['text'], 50)

word =[]
count = []
for i in words_women:
    word.append(i[0])
    count.append(i[1])
for i in words_men:
    word.append(i[0])
    count.append(i[1])

words_top_it = pd.DataFrame({'word': word, 'count':count})
words_top_it['geschlecht'] = list(np.repeat(['frau','mann'], len(words_top_n)/2))

words_top_n = words_top_n.append(words_top_fr)
words_top_n = words_top_n.append(words_top_it)
words_top_n['language'] = list(np.repeat(['Deutsch','Französisch', 'Italienisch'], len(words_top_fr)))

#### network stuff ###########
# df = pd.read_csv('C:\\Users\\tobia\\Dropbox (Privat)\\MAS_DS\\MasterThesis\\tweets_210222.csv', sep=';', low_memory=False)
# df['created_at'] = pd.to_datetime(df['created_at'])
df['addressees'] = df['addressees'].str.split(',')
df_nx = df.explode('addressees')

politicians = list(df_person['user_handle'])

df_nx['addressees'] = df_nx['addressees'].str.replace(r'\\u2069', '')
df_nx['addressees'] = df_nx['addressees'].str.replace(r'\ |\?|\.|\!|\/|\;|\:|\)|\(|\[|\]|\'|\\|\,|\...', '')
df_nx['addressees'] = df_nx['addressees'].str.replace(r'Killian_Baumanns', 'Killian_Baumann')

len(df_nx['addressees'].unique())

df_nx = df_nx[df_nx['user_handle'].str.contains('|'.join(politicians))]
df_nx = df_nx[df_nx['addressees'].str.match('|'.join(politicians))]


####
A = list(df_nx["user_handle"].unique())
B = list(df_nx["addressees"].unique())


node_list = list(set(A+B))
G = nx.DiGraph()
for i in node_list:
    G.add_node(i)
for i,j in df_nx.iterrows():
    G.add_edges_from([(j["user_handle"],j["addressees"])])

pos = nx.spring_layout(G, k= .5, iterations=50, seed=666)
# pos = nx.kamada_kawai_layout(G)



for n, p in pos.items():
    G.nodes[n]['pos'] = p


edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5,color='#888'),
    hoverinfo='none',
    mode='lines')
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])
node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='Earth',
        reversescale=True,
        color=[],
        size=15,
        colorbar=dict(
            thickness=10,
            title='# outgoing node connections',
            xanchor='left',
            titleside='right'
        ),
        line=dict(width=0)))
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])

for node, adjacencies in enumerate(G.adjacency()):
    node_trace['marker']['color']+=tuple([len(adjacencies[1])])
    node_info = adjacencies[0] +' # of outgoing connections: '+str(len(adjacencies[1]))
    node_trace['text']+=tuple([node_info])

fig_network = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='(6) Parliamentarian Twitter communication network<br><br>'
                      'Note: links represent all outgoing @mentions for all parliamentarians',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                 template='plotly_dark',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="",
                    showarrow=False,
                    xref="paper", yref="paper") ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

#### parallel categories chart ####

len(df_nx['addressees'].unique())

df_nx = df_nx[df_nx['addressees'].str.contains('|'.join(df_nx['user_handle']))]


geschlecht_empfaenger = []
partei_empfaenger = []
for i in df_nx['addressees']:
    df_i = df_nx[df_nx['user_handle'].str.startswith(i)]
    if (df_i.size != 0):
        geschlecht_empfaenger.append(df_i.loc[df_i.index[0], 'geschlecht'])
        partei_empfaenger.append(df_i.loc[df_i.index[0], 'partei'])
    else:
        geschlecht_empfaenger.append("unknown")
        partei_empfaenger.append("unknown")

df_nx['geschlecht_addressees'] = geschlecht_empfaenger
df_nx['partei_addressees'] = partei_empfaenger

gender_list = []
for i in df_nx['geschlecht_addressees']:
    if type(i)==str:
        gender_list.append(i)
    elif i is None:
        gender_list.append("unknown")
    else:
        gender_list.append(i.iloc[0])

party_list = []
for i in df_nx['partei_addressees']:
    if type(i)==str:
        party_list.append(i)
    elif i is None:
        party_list.append("unknown")
    else:
        party_list.append(i.iloc[0])
df_nx['geschlecht_addressees'] = gender_list
df_nx['partei_addressees'] = party_list

df_nx = df_nx[df_nx['geschlecht_addressees'] != 'unknown']
gender_comm = []
for i, j in zip(df_nx['geschlecht'], df_nx['geschlecht_addressees']):
    if i == j:
        gender_comm.append(0)
    else:
        gender_comm.append(1)

df_nx['gender_comm'] = gender_comm


fig_communication = px.parallel_categories(df_nx, dimensions=['geschlecht','partei',
                                                              'partei_addressees', 'geschlecht_addressees'],
                title='(7): Gender symmetry in communicative exchanges<br>breakdown of who (gender/party) talks to whom (party/gender) on Twitter.'
                      'Note: blue paths are cross-gender group and yellow paths within-gender group exchanges',
                color='gender_comm',
                color_continuous_scale= px.colors.diverging.Earth,
                template='plotly_dark',
                labels={'geschlecht':'Gender (sender)', 'partei':'Party (sender)',
                        'partei_addressees':'Party (receiver)', "geschlecht_addressees": 'Gender (receiver)',
                        'frau': 'woman', 'mann': 'man'})
fig_communication.layout.update(showlegend=False)

#### create dataset for statistical testing #####
#this dataset is also used for figure 3
eigenvector = nx.eigenvector_centrality(G)
indegree = nx.in_degree_centrality(G)
outdegree = nx.out_degree_centrality(G)
betweenness = nx.betweenness_centrality(G)
reciprocity = nx.reciprocity(G, nodes=node_list)

df_nx_centrality = pd.DataFrame.from_dict(eigenvector, orient='index', columns=['eigenvector'])
df_nx_centrality.index.name = 'user_handle'
df_nx_centrality.reset_index(inplace=True)
df_nx_centrality['indegree'] = df_nx_centrality['user_handle'].map(indegree)
df_nx_centrality['outdegree'] = df_nx_centrality['user_handle'].map(outdegree)
df_nx_centrality['betweenness'] = df_nx_centrality['user_handle'].map(betweenness)
df_nx_centrality['reciprocity'] = df_nx_centrality['user_handle'].map(reciprocity)


addressees_sentiment = df_nx.groupby([df_nx['addressees']]).mean()
addressees_sentiment.reset_index(inplace=True)
addressees_sentiment.rename(columns={"addressees": "user_handle",
                                     'polarity': 'polarity_received',
                                     'polarity_dich': 'polarity_dich_received',
                                     'gender_comm': 'gender_comm_received'}, inplace=True)
addressees_sentiment = addressees_sentiment[['user_handle', 'polarity_received', 'polarity_dich_received', 'gender_comm_received']]
gender_comm_sender = df_nx.groupby([df_nx['user_handle']]).mean()
gender_comm_sender.reset_index(inplace=True)
gender_comm_sender = gender_comm_sender[['user_handle', 'gender_comm']]

df_nx_centrality = df_nx_centrality.merge(addressees_sentiment, on = 'user_handle', how = 'left')
df_nx_centrality = df_nx_centrality.merge(gender_comm_sender, on = 'user_handle', how = 'left')
df_analysis = df_person.merge(df_nx_centrality, on = 'user_handle', how = 'right')

df_analysis = df_analysis[df_analysis['name'].notnull()]

##### main figures #####
df_days['geschlecht'].replace({'frau':'women','mann':'men'}, inplace = True)
fig_day_gender_counts = px.line(df_days,
                                x='created_at',
                                y= 'abs_count',
                                title='(2) Temporal dynamics <br><br>Twitter output (upper panel) and average sentiment (lower panel) over time',
                                color = 'geschlecht',
                                color_discrete_sequence=px.colors.qualitative.Pastel,
                                labels= {'abs_count': 'Number of posted tweets',
                                         'created_at': '',
                                         'geschlecht': 'Gender',
                                         'mann':'men',
                                         'frau':'women'},
                                hover_data= {'created_at': '|%B %d, %Y'},
                                template= 'plotly_dark')
                                #width= ,
                                #height=
fig_day_gender_counts.update_layout(xaxis= dict(tickmode = 'linear',
                                                dtick = 86400000*2))
fig_day_gender_counts.add_hline(y=df_days['abs_count'].mean(),
                                line_dash = 'dash', line_color = 'grey')
fig_day_gender_counts.update_layout(hovermode="x")

fig_day_gender_sentiment = px.bar(df_days,
                                   x='created_at',
                                   y='polarity',
                                   title='',
                                  barmode='overlay',
                                   color = 'geschlecht',
                                   color_discrete_sequence=px.colors.qualitative.Pastel,
                                   labels= {'polarity': 'Average Tweet Sentiment',
                                         'created_at': '',
                                            'geschlecht':'Gender'},
                                   hover_data={'created_at': '|%B %d, %Y'},
                                   height = 300,
                                   template='plotly_dark')
fig_day_gender_sentiment.update_layout(xaxis= dict(showticklabels=False,
                                                   dtick = 86400000*2,
                                                   tickmode = 'linear'))
fig_day_gender_sentiment.add_hline(y=df_days['polarity'].mean(),
                                line_dash = 'dash', line_color = 'grey')
fig_day_gender_sentiment.update_layout(hovermode="x")


df_analysis['geschlecht'].replace({'mann':'men','frau':'women'}, inplace = True)
df_analysis['point_size'] = np.repeat(2, len(df_analysis))
fig_person_sentiment = px.scatter(df_analysis,
                          x="polarity",
                          y="polarity_received",
                                  size="point_size",
                                  size_max=8,
                        color='geschlecht',
                        color_discrete_map={'women':'#66c5cc','men':'#f6cf71'},
                                  hover_name= "name",
                                  hover_data={'partei':True,
                                              'geschlecht':False,
                                              'point_size':False,
                                              'polarity_received':True,
                                              'polarity':True},
                          title='(3) Two-way tweet sentiment <br><br>Average sender vs. average receiver sentiment',
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        labels={'polarity': 'Average sent tweet sentiment',
                                'polarity_received':'Average received tweet sentiment',
                                  'partei':'party',
                                'geschlecht':'Gender'},
                          template='plotly_dark')
fig_person_sentiment.add_hline(y=df_analysis['polarity'].mean(),
                                line_dash = 'dash', line_color = 'grey')
fig_person_sentiment.add_vline(x=df_analysis['polarity_received'].mean(),
                                line_dash = 'dash', line_color = 'grey')

df_person['geschlecht'].replace({'mann':'men','frau':'women'}, inplace = True)
fig_person_all = px.treemap(df_person,
                            path=[px.Constant('All parliamentarians'), 'partei', 'name'],
                            values='abs_count',
                            color='geschlecht',
                            hover_name='name',
                            color_discrete_map={'(?)':'black','women':'#66c5cc','men':'#f6cf71'},
                          title="(1) The 'Twitterverse' <br><br>Breakdown of tweet volume of Swiss parliamentarians per gender/party)",
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            labels={'abs_count': '# posted tweets',
                                  'partei':'party',
                                    'geschlecht':'Gender'},
                          template='plotly_dark')

fig_person_retweet = px.scatter(df_person,
                          x="abs_count",
                          y="polarity",
                        color='geschlecht',
                        size='retweet_count',
                        size_max= 63,
                          title='(4) Output-throughput-input<br><br> Posted tweets (x-axis) vs. sender sentiment (y-axis) vs. received mentions (size)',
                                hover_name= 'name',
                                hover_data={'partei':True,
                                            'geschlecht':False},
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        labels={'polarity': 'Average sender tweet sentiment',
                                  'abs_count':'Number of posted tweets',
                                'partei':'party',
                                'geschlecht': 'Gender','retweet_count': 'Number of received @s (size)'},
                          template='plotly_dark')
fig_person_retweet.add_hline(y=df_person['polarity'].mean(),
                                line_dash = 'dash', line_color = 'grey')
fig_person_retweet.add_vline(x=df_person['abs_count'].mean(),
                                line_dash = 'dash', line_color = 'grey')

words_top_n['geschlecht'].replace({'frau':'women','mann':'men'}, inplace = True)
words_top_n['language'].replace({'Deutsch':'German','Französisch':'French',
                                 'Italienisch': 'Italian'}, inplace = True)

diff_de = words_top_n[words_top_n['language']=='German'].pivot_table(index='word', columns = 'geschlecht', values = 'count').reset_index().fillna(0)
diff_de['language'] = np.repeat('German', len(diff_de))
diff_fr = words_top_n[words_top_n['language']=='French'].pivot_table(index='word', columns = 'geschlecht', values = 'count').reset_index().fillna(0)
diff_fr['language'] = np.repeat('French', len(diff_fr))
diff_it = words_top_n[words_top_n['language']=='Italian'].pivot_table(index='word', columns = 'geschlecht', values = 'count').reset_index().fillna(0)
diff_it['language'] = np.repeat('Italian', len(diff_it))

words_diff = diff_de.append(diff_fr).append(diff_it)
words_diff['diff'] = (words_diff['women']-words_diff['men'])
def diff_dir(series):
    if series <=0:
        return 'Men more'
    elif series > 0:
        return 'Women more'
words_diff['diff_dir'] = words_diff['diff'].apply(diff_dir)
words_diff.sort_values(by='diff', ascending = True, inplace=True)


fig_word_diff_de = px.bar(words_diff[words_diff['language']=='German'],
                       x='diff',
                       y='word',
                       color='diff_dir',
                       color_discrete_map={'Women more': '#66c5cc', 'Men more': '#f6cf71'},
                       # facet_col='language',
                       template='plotly_dark',
                       title="(5) Parliament language <br><br> Gender difference in the 50 most frequently used German words",
                       labels= {'language': 'Language',
                 'diff': 'Gender difference in word count',
                    'word': 'Words',
                    'diff_dir':'Direction of difference'},
                       color_discrete_sequence=px.colors.qualitative.Pastel,
                       hover_name='word',
                          height=850,
                          width=550)
fig_word_diff_de.update_layout(yaxis= dict(dtick=1))
fig_word_diff_de.update_layout(hovermode="y unified")
fig_word_diff_de.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-.14,
    xanchor="right",
    x=1))



fig_word_diff_fr = px.bar(words_diff[words_diff['language']=='French'],
                       x='diff',
                       y='word',
                       color='diff_dir',
                       color_discrete_map={'Women more': '#66c5cc', 'Men more': '#f6cf71'},
                       # facet_col='language',
                       template='plotly_dark',
                       title="<br><br>Gender difference in top 50 used French words",
                       labels= {'language': 'Language',
                 'diff': 'Gender difference in word count',
                    'word': 'Words',
                    'diff_dir':'Direction of difference'},
                       color_discrete_sequence=px.colors.qualitative.Pastel,
                       hover_name='word',
                          height=850,
                          width=550)
fig_word_diff_fr.update_layout(yaxis= dict(dtick=1))
fig_word_diff_fr.update_layout(hovermode="y unified")
fig_word_diff_fr.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-.14,
    xanchor="right",
    x=1))


# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("Overview of Swiss Parliamentarian Twitter Behavior", style={'text-align': 'center'}),
    dcc.Textarea(id='intro_text',
                 cols=2,
                 value='This dashboard provides an near-realtime* analyisis of'
                       ' Swiss parliamentarians Twitter communication.\n'
                       'The goal of the dashboard is to visualize gender differences/similarities across'
                       'a range of metrics. As a general rule, data referring to women parliamentarians is'
                       ' colored blue while yellow represents men parliamentarians. Most visualizations are interactive,'
                       ' i. e., more information can be learned by hovering over, clicking on, or selecting data points.\n'
                       'The data underlying these visualizations are all tweets that are either posted by parliamentarians or that'
                       ' mention parliamentarians.\n'
                       '\n*This deployed version does not update the data because of technical constraints.'
                       'The code can be tweaked to incorporate automatic updates on local machnies',
                 title= 'INTRODUCTION',
                 style={'width':'100%',
                        'height':'140px',
                        'backgroundColor': '#111111', #ev muss dies noch schwärzer werden
                        'color':'white'}),
    dcc.Graph(id='fig_person_all', figure=fig_person_all),
    dcc.Graph(id='fig_day_gender_counts', figure=fig_day_gender_counts),
    dcc.Graph(id='fig_day_gender_sentiment', figure=fig_day_gender_sentiment),
    dcc.Graph(id= 'fig_person_sentiment', figure=fig_person_sentiment),
    dcc.Graph(id='fig_person_retweet', figure = fig_person_retweet),
    html.Div(children=[
        dcc.Graph(id='fig_word_diff_de', figure=fig_word_diff_de, style={'display':'inline-block',
                                                                         'backgroundColor': '#111111'}),
        dcc.Graph(id='fig_word_diff_fr', figure=fig_word_diff_fr, style={'display':'inline-block',
                                                                         'backgroundColor': '#111111'})
    ]),
    dcc.Graph(id='fig_network', figure= fig_network),
    dcc.Graph(id='fig_communication', figure=fig_communication),
    dcc.Textarea(id='outro-text',
                 cols=2,
                 value=
                       '\n Note: This dashboard is part of a master thesis in data science. The full documentation and code can be'
                       'found at: https://github.com/RohrbaTo/parliament-twitter \n'
                       '\nTobias Rohrbach\n'
                       'contact: tobiasrohrbach@hotmail.com',
                 title='',
                 style={'width': '100%',
                        'height': '100px',
                        'backgroundColor': '#111111',  # ev muss dies noch schwärzer werden
                        'color': 'white'})
])


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=False)
