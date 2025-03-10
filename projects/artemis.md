---
layout: project
title: 'Artemis'
---

## About the project

**Artemis Project: Mapping Historical Landscapes for Environmental Restoration**

Restoring ecosystems is a pressing global challenge, yet determining what to restore requires a deep understanding of historical landscapes. Over the past centuries, human activities have drastically transformed natural environments, making it difficult to define a single benchmark for restoration. Instead, Artemis aims to reconstruct past landscapes using high-resolution historical maps to provide insights into long-term environmental changes. Belgium, with its rich cartographic heritage, serves as an ideal case study. By leveraging cutting-edge digital humanities techniques, Artemis will develop an open data infrastructure for analyzing historical maps and landscapes, with a primary focus on the River Scheldt Valley—one of Europe’s most historically significant and densely populated regions.

Historical maps are invaluable resources that not only depict past landscapes but also reflect human-environment interactions. However, these maps remain underutilized due to the complexity of extracting structured data from them. Artemis seeks to overcome this challenge by combining geospatial technologies, machine learning, and natural language processing (NLP) to unlock the potential of historical maps for environmental research. This project will facilitate the automatic extraction of key landscape features, enabling researchers to analyze long-term land use changes and support ecological restoration initiatives.

## IDLab role
To achieve these goals, IDLab will develop advanced processing pipelines for extracting and enriching historical map data. Key tasks include:

#### 1. Text Recognition & Geocoding
- **MapReader Integration:** Each historical map will undergo text recognition processing using the state-of-the-art [MapReader](https://mapreader.readthedocs.io/) tool to identify place names and locations.
- **Geocoding Historical Toponyms:** Extracted place names will be geocoded using contemporary and historical data from the Belgian Historical Gazetteer. This will help estimate the map geolocation and facilitate spatial analysis.
- **Toponym-Based Geolocation:** IDLab will apply machine learning techniques to estimate map geolocation based on extracted toponyms, building upon previously developed methods.

![Text Recognition](/assets/img/projects/ARTEMIS/artemis_crop.png "Text Recognition")


#### 2. Extraction of visual features
- **Cadastre Map Parsing** For the Primitive and Reduced Cadastres, IDLab will develop custom line and color-based detection methods, to segment each parcel. We will combine this with the text recognition results and link each parcel to its unique identifier.
- **Automated Road & Feature Extraction:** Most of the heterogeneous handwritten maps contain visually distinctive roads or features that can be automatically extracted using a semi-automatic segmentation pipeline. 
- **High-Density Map Processing:** The georeferenced topographic maps of Vandermaelen and Ferraris are characterised by a high
density of information. While colour and texture-based methods could be used to estimate land use,
they may yield lower accuracy. Therefore, we will train a custom land use segmentation model. Additionally, new approaches such as domain adaptation (Wu et al. 2023) and synthetic
data generation will be investigated to leverage existing labelled datasets.

Through these innovations, Artemis will transform historical cartographic archives into structured geospatial data, bridging the gap between environmental history and modern restoration efforts. By integrating state-of-the-art digital humanities and environmental research techniques, the project will provide a crucial reference framework for future ecological rehabilitation projects.




