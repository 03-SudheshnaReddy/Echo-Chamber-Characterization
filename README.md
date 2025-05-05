# Echo-Chamber-Characterization
# Social Network Analysis: Political Bias Detection & Echo Chamber Simulation

This project simulates a social network of 36 users with varying political leanings, activity levels, and tweet biases. The main objective is to analyze user interactions, detect political bias using production polarity and variance, and examine how different political groups interact in a social network. The project includes two models for comparison: a **Baseline Model** and a **Power Law Model**.

### Key Metrics and Concepts
- **Production Polarity**: This measures the average political bias in a user's tweets. The polarity can be categorized into:
  - **Government**: 1
  - **Opposition**: 0
  - **Neutral**: 0.5

- **Production Variance**: Measures the consistency of a user's political opinions across their tweets. Low variance indicates consistent views, while high variance indicates diverse political opinions.

- **Mixing Patterns**: This metric analyzes how much users interact with others who share the same political views (homophily) versus different views (heterophily). A high mixing pattern value indicates homophily (similar opinions), while a low value indicates heterophily (diverse opinions).

- **EI Index**: The **External-Internal (EI) Index** measures the strength of interactions within and across political groups:
  - **Homophily**: More internal connections (EI < 0)
  - **Neutral**: Equal internal and external connections (EI = 0)
  - **Heterophily**: More cross-ideological interactions (EI > 0)

### Models
1. **Baseline Model**
   - **Adjacency Matrix**: A random 36x36 matrix is created where each entry (0 or 1) indicates whether two users are connected.
   - **Mixing Patterns**: For each user, we calculate the ratio of same-group connections to total connections. The result is averaged to determine the group's overall mixing pattern.
   - **Political Labels**: Users are randomly assigned political leanings (Government, Opposition, or Neutral). Activity levels (Low, Medium, High) are randomly assigned.
   - **EI Index**: For each group (Government, Opposition, Neutral), we compute the EI index to evaluate the level of homophily or heterophily.
   
2. **Power Law Model**
   - **Scale-Free Network**: This model generates a scale-free network based on a power-law degree distribution. Most users have a few connections, while a few "hub" users have many.
   - **Preferential Attachment**: New users are more likely to connect to users with many connections, similar to real-world social media dynamics.
   - **Political Labels**: Political leanings are assigned based on the users' activity levels to avoid bias, following a sorted order.
   - **EI Index**: Similar to the Baseline Model, we calculate the EI index to measure the strength of interactions within and between political groups.

### Dataset
- **Baseline Model**: Political leanings are randomly assigned, with activity levels classified as Low, Medium, or High.
- **Power Law Model**: Users are sorted by their activity levels, and political leanings (Government, Opposition, or Neutral) are assigned according to the sorted order. This approach avoids bias by ensuring that high-activity users are not overrepresented in any particular group.

### Steps to Run the Code

1. **Install Required Libraries**:
   - `networkx`: For graph creation and network analysis.
   - `numpy`: For numerical operations.
   - `pandas`: For data manipulation.
   - `matplotlib`: For visualizing the results.

   ```bash
   pip install networkx numpy pandas matplotlib
# Political Bias and Echo Chamber Detection in Social Networks

## Run the Code

- **Generate a random adjacency matrix** for the **Baseline Model** using a uniform connection probability.
- **Create a power-law-based adjacency matrix** for the **Power Law Model** using the Barabási–Albert preferential attachment algorithm.
- **Calculate key metrics** for both models:
  - **Production Polarity**: Measures the average political leaning of users' content.
  - **Variance of Production Polarity**: Shows how diverse the content is within political groups.
  - **Mixing Patterns**: Analyzes interactions between different political groups.
  - **EI Index**: Evaluates the ratio of internal to external connections for each group to detect echo chamber tendencies.

## View Results

- The program outputs the **average** and **standard deviation** of mixing patterns for each political group.
- It also reports the **number of internal and external connections** and computes the **EI Index**.
- Based on the EI Index, each political group is classified as:
  - **Homophilic**: Prefers interacting within its own group.
  - **Neutral**: Interacts equally with others.
  - **Heterophilic**: Interacts more with different groups.
- Results are shown for **both models**, helping visualize how different network structures affect group interactions and potential echo chamber formation.

## Output:
1.BaseLine model
![baseline_model_graph](https://github.com/user-attachments/assets/ca66067a-a4f3-40b0-8f26-2e2bc52a581a)
![image](https://github.com/user-attachments/assets/6a2315e9-ef3b-43b5-ac83-c7d868e5d266)
![image](https://github.com/user-attachments/assets/c4cc7592-ab1b-4659-8e4e-33f69001ea56)

2.PowerLaw model
![image](https://github.com/user-attachments/assets/40fb0a6b-07ff-4be0-8ba8-402a59ba55b4)
![image](https://github.com/user-attachments/assets/722c5f7d-27b2-4268-a9d1-e1c8aa04ad8f)

## Conclusion
This project provides valuable insights into political bias and interaction patterns within a simulated social network. By using both a baseline model and a scale-free power law model, it offers a comprehensive view of how political groups interact in a network. The findings can be used to analyze the formation of echo chambers and polarization in online social networks.

## Future Work
- **Extended Dataset**: The model can be expanded to include more users and different political labels.
- **Incorporate More Metrics**: Additional metrics like tweet frequency, sentiment analysis, or geographic location could further enrich the analysis.
- **Real-World Application**: The model could be applied to real-world social networks or news aggregator platforms to study political polarization and echo chambers.

## SEE REPORT ATTACHED ABOVE FOR COMPLETE INFORMATION OF PROJECT
