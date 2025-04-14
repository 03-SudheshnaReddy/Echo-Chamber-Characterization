# import pandas as pd
# import random
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# import textwrap

# # User Data Generation
# NUM_USERS = 36
# LOCATIONS = ["USA", "India"]
# MACRO_TOPICS = ["Politics", "Sports", "Entertainment", "Technology"]
# MICRO_TOPICS = ["Politics", "Sports", "Entertainment", "Technology"]
# POLITICAL_LEANINGS = ["Government", "Opposition", "Neutral"]
# ACTIVITY_LEVELS = ["Low Active", "Medium Active", "High Active"]
# TWEET_POLARITY_VALUES = {"Government": 1, "Opposition": 0, "Neutral": 0.5}

# def assign_activity_level():
#     return random.choice(ACTIVITY_LEVELS)

# def assign_political_leaning():
#     return random.choice(POLITICAL_LEANINGS)

# def generate_tweet_polarities(political_leaning, num_tweets):
#     if political_leaning == "Government":
#         gov_percent = random.uniform(0.7, 1.0)
#         neutral_percent = random.uniform(0, 1 - gov_percent)
#         opp_percent = 1 - gov_percent - neutral_percent
#     elif political_leaning == "Opposition":
#         opp_percent = random.uniform(0.7, 1.0)
#         neutral_percent = random.uniform(0, 1 - opp_percent)
#         gov_percent = 1 - opp_percent - neutral_percent
#     else:
#         neutral_percent = random.uniform(0.6, 1.0)
#         gov_percent = random.uniform(0, 1 - neutral_percent)
#         opp_percent = 1 - neutral_percent - gov_percent

#     num_gov = int(round(gov_percent * num_tweets))
#     num_opp = int(round(opp_percent * num_tweets))
#     num_neutral = num_tweets - num_gov - num_opp

#     tweet_polarities = (
#         [TWEET_POLARITY_VALUES["Government"]] * num_gov +
#         [TWEET_POLARITY_VALUES["Opposition"]] * num_opp +
#         [TWEET_POLARITY_VALUES["Neutral"]] * num_neutral
#     )
#     random.shuffle(tweet_polarities)
#     return tweet_polarities

# def calculate_production_polarity(tweet_polarities):
#     return np.mean(tweet_polarities)

# def calculate_production_variance(tweet_polarities):
#     return np.var(tweet_polarities)

# def is_delta_partisan(production_polarity, delta=0.2):
#     if production_polarity <= delta:
#         return "Biased (Opposition)"
#     elif production_polarity >= 1 - delta:
#         return "Biased (Government)"
#     else:
#         return "Not Biased"

# # Generate user data
# data = []
# activity_levels_pool = ACTIVITY_LEVELS * (NUM_USERS // 3)
# political_leanings_pool = POLITICAL_LEANINGS * (NUM_USERS // 3)
# random.shuffle(activity_levels_pool)
# random.shuffle(political_leanings_pool)

# for user_id in range(1, NUM_USERS + 1):
#     location = random.choice(LOCATIONS)
#     political_leaning = political_leanings_pool.pop()
#     activity_level = activity_levels_pool.pop()

#     if activity_level == "Low Active":
#         num_tweets = random.randint(1, 5)
#     elif activity_level == "Medium Active":
#         num_tweets = random.randint(6, 10)
#     else:
#         num_tweets = random.randint(11, 20)

#     tweet_polarities = generate_tweet_polarities(political_leaning, num_tweets)
#     production_polarity = calculate_production_polarity(tweet_polarities)
#     production_variance = calculate_production_variance(tweet_polarities)
#     delta_partisan_status = is_delta_partisan(production_polarity)

#     macro_topic = random.choice(MACRO_TOPICS)
#     micro_topic = random.choice([t for t in MICRO_TOPICS if t != macro_topic])

#     data.append({
#         "User ID": f"U{user_id}",
#         "Location": location,
#         "Political Leaning": political_leaning,
#         "Activity Level": activity_level,
#         "Macro Topic": macro_topic,
#         "Micro Topic": micro_topic,
#         "Number of Tweets": num_tweets,
#         "Production Polarity": production_polarity,
#         "Production Variance": production_variance,
#         "δ-Partisan (δ=0.2)": delta_partisan_status
#     })

# df = pd.DataFrame(data)

# # ✅ Scale-free network generation
# def generate_scale_free_network(n_users, m=2, p=0.0):
#     # Using powerlaw_cluster_graph which supports triangle formation
#     G = nx.powerlaw_cluster_graph(n_users, m=m, p=p, seed=42)
#     adj = nx.to_numpy_array(G, dtype=int)
#     return adj

# # Graph creation
# def create_graph(adj_matrix, user_data):
#     G = nx.from_numpy_array(adj_matrix)
#     for idx, row in user_data.iterrows():
#         for col in user_data.columns:
#             G.nodes[idx][col] = row[col]
#     return G




# def calculate_mixing_patterns(G):
#     mixing = {}
#     for node in G.nodes():
#         neighbors = list(G.neighbors(node))
#         if not neighbors:
#             mixing[node] = 0
#             continue
            
#         node_leaning = G.nodes[node]['Political Leaning']
#         total_score = 0
        
#         for n in neighbors:
#             neighbor_leaning = G.nodes[n]['Political Leaning']
            
       
#             if neighbor_leaning == node_leaning:
#                 total_score += 1
            
#             elif (node_leaning == "Government" and neighbor_leaning == "Opposition") or \
#                  (node_leaning == "Opposition" and neighbor_leaning == "Government"):
#                 total_score -= 1
  
#             else:
#                 total_score += 0
        
      
#         mixing[node] = total_score / len(neighbors) if neighbors else 0
        
#     return mixing

# def calculate_ei_index(G):
#     inter_group_edges = 0
#     intra_group_edges = 0
#     for edge in G.edges():
#         node1, node2 = edge
#         if G.nodes[node1]['Political Leaning'] != G.nodes[node2]['Political Leaning']:
#             inter_group_edges += 1
#         else:
#             intra_group_edges += 1
#     total = inter_group_edges + intra_group_edges
#     if total == 0:
#         return {'EI Index': 0, 'Inter-group Edges': 0, 'Intra-group Edges': 0}
#     ei_index = (inter_group_edges - intra_group_edges) / total
#     return {'EI Index': ei_index, 'Inter-group Edges': inter_group_edges, 'Intra-group Edges': intra_group_edges}

# def calculate_group_ei_indices(G):
#     groups = list(set(nx.get_node_attributes(G, 'Political Leaning').values()))
#     ei_results = {}
#     for group in groups:
#         inter_group_edges = 0
#         intra_group_edges = 0
#         group_nodes = [n for n in G.nodes if G.nodes[n]['Political Leaning'] == group]
#         for node in group_nodes:
#             for neighbor in G.neighbors(node):
#                 if G.nodes[neighbor]['Political Leaning'] == group:
#                     intra_group_edges += 1
#                 else:
#                     inter_group_edges += 1
#         intra_group_edges //= 2
#         inter_group_edges //= 2
#         total_edges = inter_group_edges + intra_group_edges
#         ei_index = (inter_group_edges - intra_group_edges) / total_edges if total_edges != 0 else 0
#         ei_results[group] = {
#             'EI Index': ei_index,
#             'Inter-group Edges': inter_group_edges,
#             'Intra-group Edges': intra_group_edges,
#             'Total Edges': total_edges
#         }
#     return ei_results

# def print_wrapped(text, width=120):
#     wrapped_text = textwrap.fill(text, width=width)
#     print(wrapped_text)

# # Main Execution
# adj_matrix = generate_scale_free_network(NUM_USERS, m=2, p=0.0)
# G = create_graph(adj_matrix, df)
# mixing_patterns = calculate_mixing_patterns(G)

# nx.set_node_attributes(G, mixing_patterns, 'Mixing Pattern')
# df['Mixing Pattern'] = [mixing_patterns[i] for i in range(len(df))]

# ei_results = calculate_ei_index(G)
# group_ei_results = calculate_group_ei_indices(G)

# # Visualization
# plt.figure(figsize=(20, 15))
# color_map = {"Government": "red", "Opposition": "blue", "Neutral": "green"}
# colors = [color_map[G.nodes[node]['Political Leaning']] for node in G.nodes]
# pos = nx.spring_layout(G, seed=42)

# size_map = {"Low Active": 200, "Medium Active": 400, "High Active": 600}
# sizes = [size_map[G.nodes[node]['Activity Level']] for node in G.nodes]

# nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.9)
# nx.draw_networkx_edges(G, pos, alpha=0.2)
# nx.draw_networkx_labels(G, pos, font_size=8)

# for node in G.nodes:
#     x, y = pos[node]
#     info = (f"M:{G.nodes[node]['Mixing Pattern']:.2f}\n"
#             f"P:{G.nodes[node]['Production Polarity']:.2f}\n"
#             f"δ:{G.nodes[node]['δ-Partisan (δ=0.2)'][:1]}")
#     plt.text(x, y + 0.07, info, ha='center', va='center', fontsize=7,
#              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# for leaning, color in color_map.items():
#     plt.scatter([], [], c=color, label=f"{leaning}")
# plt.legend(title="Political Leaning")
# plt.title("Scale-Free Social Network\nNode size = Activity Level | M = Mixing pattern | P = Production polarity | δ = Partisan status")
# plt.axis('off')

# # Console Output
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_colwidth', None)

# print("\n" + "="*120)
# print_wrapped("COMPLETE USER DATA")
# print("="*120)
# print(df)

# print("\n" + "="*120)
# print_wrapped("ADJACENCY MATRIX")
# print("="*120)
# print(adj_matrix)

# print("\n" + "="*120)
# print_wrapped("MIXING PATTERN STATISTICS")
# print("="*120)
# print(df.groupby('Political Leaning')['Mixing Pattern'].describe())

# print("\n" + "="*120)
# print_wrapped("EI INDEX ANALYSIS")
# print("="*120)

# print("\nOverall Network:")
# for key, value in ei_results.items():
#     print_wrapped(f"{key}: {value}")

# print("\nGroup-Specific Analysis:")
# for group, results in group_ei_results.items():
#     print(f"\n{group} Group:")
#     for key, value in results.items():
#         print_wrapped(f"  {key}: {value}")
#     interpretation = "Homophily" if results['EI Index'] < 0 else "Heterophily"
#     strength = "strong" if abs(results['EI Index']) > 0.5 else "moderate" if abs(results['EI Index']) > 0.2 else "weak"
#     print_wrapped(f"  Interpretation: {interpretation} ({strength})")

# plt.tight_layout()
# plt.show()

import pandas as pd
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import textwrap
from collections import defaultdict

# Constants
NUM_USERS = 36
LOCATIONS = ["USA", "India"]
MACRO_TOPICS = ["Politics", "Sports", "Entertainment", "Technology"]
MICRO_TOPICS = ["Politics", "Sports", "Entertainment", "Technology"]
POLITICAL_LEANINGS = ["Government", "Opposition", "Neutral"]
ACTIVITY_LEVELS = ["Low Active", "Medium Active", "High Active"]
TWEET_POLARITY_VALUES = {"Government": 1, "Opposition": 0, "Neutral": 0.5}
ECHO_THRESHOLD = 0.5  # Clustering coefficient threshold

# Helper Functions
def assign_activity_level():
    return random.choice(ACTIVITY_LEVELS)

def generate_tweet_polarities(political_leaning, num_tweets):
    if political_leaning == "Government":
        gov_percent = random.uniform(0.7, 1.0)
        neutral_percent = random.uniform(0, 1 - gov_percent)
        opp_percent = 1 - gov_percent - neutral_percent
    elif political_leaning == "Opposition":
        opp_percent = random.uniform(0.7, 1.0)
        neutral_percent = random.uniform(0, 1 - opp_percent)
        gov_percent = 1 - opp_percent - neutral_percent
    else:
        neutral_percent = random.uniform(0.6, 1.0)
        gov_percent = random.uniform(0, 1 - neutral_percent)
        opp_percent = 1 - neutral_percent - gov_percent

    num_gov = int(round(gov_percent * num_tweets))
    num_opp = int(round(opp_percent * num_tweets))
    num_neutral = num_tweets - num_gov - num_opp

    tweet_polarities = (
        [TWEET_POLARITY_VALUES["Government"]] * num_gov +
        [TWEET_POLARITY_VALUES["Opposition"]] * num_opp +
        [TWEET_POLARITY_VALUES["Neutral"]] * num_neutral
    )
    random.shuffle(tweet_polarities)
    return tweet_polarities

def calculate_production_polarity(tweet_polarities):
    return np.mean(tweet_polarities)

def calculate_production_variance(tweet_polarities):
    return np.var(tweet_polarities)

def is_delta_partisan(production_polarity, delta=0.2):
    if production_polarity <= delta:
        return "Biased (Opposition)"
    elif production_polarity >= 1 - delta:
        return "Biased (Government)"
    else:
        return "Not Biased"

# Network Generation
def generate_scale_free_network(n_users, m=2, p=0.0):
    return nx.powerlaw_cluster_graph(n_users, m=m, p=p, seed=42)

def assign_political_leanings_by_degree(G):
    degrees = sorted(G.degree(), key=lambda x: -x[1])  # Sort by degree (high to low)
    leanings_cycle = ["Government", "Opposition", "Neutral"]
    
    for i, (node, degree) in enumerate(degrees):
        leaning = leanings_cycle[i % 3]
        G.nodes[node]["Political Leaning"] = leaning
        G.nodes[node]["Degree"] = degree
    
    return G

# User Data Generation
def generate_user_data(G):
    data = []
    activity_levels_pool = ACTIVITY_LEVELS * (NUM_USERS // 3)
    random.shuffle(activity_levels_pool)

    for node in G.nodes():
        location = random.choice(LOCATIONS)
        political_leaning = G.nodes[node]["Political Leaning"]
        activity_level = activity_levels_pool.pop()
        degree = G.nodes[node]["Degree"]

        if activity_level == "Low Active":
            num_tweets = random.randint(1, 5)
        elif activity_level == "Medium Active":
            num_tweets = random.randint(6, 10)
        else:
            num_tweets = random.randint(11, 20)

        tweet_polarities = generate_tweet_polarities(political_leaning, num_tweets)
        production_polarity = calculate_production_polarity(tweet_polarities)
        production_variance = calculate_production_variance(tweet_polarities)
        delta_partisan_status = is_delta_partisan(production_polarity)

        macro_topic = random.choice(MACRO_TOPICS)
        micro_topic = random.choice([t for t in MICRO_TOPICS if t != macro_topic])

        data.append({
            "User ID": f"U{node + 1}",
            "Location": location,
            "Political Leaning": political_leaning,
            "Activity Level": activity_level,
            "Degree": degree,
            "Macro Topic": macro_topic,
            "Micro Topic": micro_topic,
            "Number of Tweets": num_tweets,
            "Production Polarity": production_polarity,
            "Production Variance": production_variance,
            "δ-Partisan (δ=0.2)": delta_partisan_status
        })

    return pd.DataFrame(data)

def create_graph(G, user_data):
    for idx, row in user_data.iterrows():
        node = int(row["User ID"][1:]) - 1
        for col in user_data.columns:
            if col != "User ID":
                G.nodes[node][col] = row[col]
    return G

# Analysis Functions
def calculate_mixing_patterns(G):
    mixing = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if not neighbors:
            mixing[node] = 0
            continue
        node_leaning = G.nodes[node]['Political Leaning']
        total_score = 0
        for n in neighbors:
            neighbor_leaning = G.nodes[n]['Political Leaning']
            if neighbor_leaning == node_leaning:
                total_score += 1
            elif (node_leaning == "Government" and neighbor_leaning == "Opposition") or \
                 (node_leaning == "Opposition" and neighbor_leaning == "Government"):
                total_score -= 1
        mixing[node] = total_score / len(neighbors) if neighbors else 0
    return mixing

def calculate_ei_index(G):
    inter_group_edges = 0
    intra_group_edges = 0
    for edge in G.edges():
        node1, node2 = edge
        if G.nodes[node1]['Political Leaning'] != G.nodes[node2]['Political Leaning']:
            inter_group_edges += 1
        else:
            intra_group_edges += 1
    total = inter_group_edges + intra_group_edges
    if total == 0:
        return {'EI Index': 0, 'Inter-group Edges': 0, 'Intra-group Edges': 0}
    ei_index = (inter_group_edges - intra_group_edges) / total
    return {'EI Index': ei_index, 'Inter-group Edges': inter_group_edges, 'Intra-group Edges': intra_group_edges}

def calculate_group_ei_indices(G):
    groups = list(set(nx.get_node_attributes(G, 'Political Leaning').values()))
    ei_results = {}
    for group in groups:
        inter_group_edges = 0
        intra_group_edges = 0
        group_nodes = [n for n in G.nodes if G.nodes[n]['Political Leaning'] == group]
        for node in group_nodes:
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['Political Leaning'] == group:
                    intra_group_edges += 1
                else:
                    inter_group_edges += 1
        intra_group_edges //= 2
        inter_group_edges //= 2
        total_edges = inter_group_edges + intra_group_edges
        ei_index = (inter_group_edges - intra_group_edges) / total_edges if total_edges != 0 else 0
        ei_results[group] = {
            'EI Index': ei_index,
            'Inter-group Edges': inter_group_edges,
            'Intra-group Edges': intra_group_edges,
            'Total Edges': total_edges
        }
    return ei_results

def identify_echo_chambers(G, threshold=ECHO_THRESHOLD):
    clustering = nx.clustering(G)
    nx.set_node_attributes(G, clustering, 'Clustering Coefficient')
    echo_chambers = {}
    for node in G.nodes():
        partisan = G.nodes[node]['δ-Partisan (δ=0.2)'].startswith("Biased")
        if partisan and clustering[node] >= threshold:
            echo_chambers[node] = True
        else:
            echo_chambers[node] = False
    nx.set_node_attributes(G, echo_chambers, 'In Echo Chamber')
    return clustering, echo_chambers

def print_wrapped(text, width=120):
    wrapped_text = textwrap.fill(text, width=width)
    print(wrapped_text)

# Main Execution
G = generate_scale_free_network(NUM_USERS, m=2, p=0.0)
G = assign_political_leanings_by_degree(G)
df = generate_user_data(G)
G = create_graph(G, df)

mixing_patterns = calculate_mixing_patterns(G)
nx.set_node_attributes(G, mixing_patterns, 'Mixing Pattern')
df['Mixing Pattern'] = [mixing_patterns[i] for i in range(len(df))]

ei_results = calculate_ei_index(G)
group_ei_results = calculate_group_ei_indices(G)
clustering_coefficients, echo_chamber_flags = identify_echo_chambers(G)

df['Clustering Coefficient'] = df.index.map(clustering_coefficients)
df['In Echo Chamber'] = df.index.map(echo_chamber_flags)

# Visualization
plt.figure(figsize=(20, 15))
color_map = {"Government": "red", "Opposition": "blue", "Neutral": "green"}
colors = [color_map[G.nodes[node]['Political Leaning']] for node in G.nodes]
pos = nx.spring_layout(G, seed=42)

degrees = [G.nodes[node]['Degree'] for node in G.nodes]
min_degree, max_degree = min(degrees), max(degrees)
sizes = [300 + 1200 * (d - min_degree)/(max_degree - min_degree) for d in degrees]

nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.9)
nx.draw_networkx_edges(G, pos, alpha=0.2)
nx.draw_networkx_labels(G, pos, font_size=8)

for node in G.nodes:
    x, y = pos[node]
    info = (f"D:{G.nodes[node]['Degree']}\n"
            f"M:{G.nodes[node]['Mixing Pattern']:.2f}\n"
            f"P:{G.nodes[node]['Production Polarity']:.2f}\n"
            f"δ:{G.nodes[node]['δ-Partisan (δ=0.2)'][:1]}\n"
            f"C:{G.nodes[node]['Clustering Coefficient']:.2f}")
    plt.text(x, y + 0.07, info, ha='center', va='center', fontsize=7,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

for leaning, color in color_map.items():
    plt.scatter([], [], c=color, label=f"{leaning}")
plt.legend(title="Political Leaning")
plt.title("Scale-Free Social Network\nNode size = Degree | M = Mixing pattern | P = Production polarity | δ = Partisan status | C = Clustering")
plt.axis('off')

# Console Output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

print("\n" + "="*120)
print_wrapped("COMPLETE USER DATA")
print("="*120)
print(df.sort_values('Degree', ascending=False))

print("\n" + "="*120)
print_wrapped("ECHO CHAMBER USERS (Clustering ≥ 0.5 AND Partisan)")
print("="*120)
print(df[df['In Echo Chamber'] == True][['User ID', 'Political Leaning', 'δ-Partisan (δ=0.2)', 'Clustering Coefficient']])

print("\n" + "="*120)
print_wrapped("EI INDEX ANALYSIS")
print("="*120)

print("\nOverall Network:")
for key, value in ei_results.items():
    print_wrapped(f"{key}: {value}")

print("\nGroup-Specific Analysis:")
for group, results in group_ei_results.items():
    print(f"\n{group} Group:")
    for key, value in results.items():
        print_wrapped(f"  {key}: {value}")
    interpretation = "Homophily" if results['EI Index'] < 0 else "Heterophily"
    strength = "strong" if abs(results['EI Index']) > 0.5 else "moderate" if abs(results['EI Index']) > 0.2 else "weak"
    print_wrapped(f"  Interpretation: {interpretation} ({strength})")

plt.tight_layout()
plt.show()