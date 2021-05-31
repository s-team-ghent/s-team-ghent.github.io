---
layout: project
title: 'UGESCO'
---

## About the project

The Ugesco project has develop geo-temporal (meta)data extraction and enrichment tools to extend and link the existing collection items and facilitate spatio-temporal collection mapping for interactive querying. In order to optimize the quality of the temporal and spatial annotations that are retrieved by our automatic enrichment tools, the UGESCO project investegated the added value of microtask crowdsourcing in validating and improving the generated metadata.

A large collection of images from the region around Brussels taken from the archive of [Cegesoma](https://www.cegesoma.be/) was analyzed with computer vision tools. First, the objects visible on the images were automatically detected and imported as additional metadata. Next, each image description was analyzed with natural language processing tools in order to extract the mentioned entities, locations, and dates. Using these entities, the image was geolocated and a streetview was suggested showing the same location where the picture was taken. Finally, using both visual and textual similarity, similar images in the collection can be found, which often depict the same location or landmark. 

A web-based crowdsourcing platform named [Ugescrowd](http://tw06v074.ugent.be/) was built. Using this platform, users can validate the automatically generated metadata in an interactive application. The collection can also be queried based on the extracted metadata, allowing for more detailed querying and a better user experience.

## IDLab role

IDLab had the following tasks within the UGESCO project

1. Automated metadata extraction using computer vision tools
2. Determinining the place and date the picture was taken using NLP techniques
3. Computational Rephotography, repeating the photograph of the same place from the same angle at different time stamps
4. Development of an interactive crowdsourcing platform to query the collection and validate the generated metadata
5. Publication: [UGESCO - A Hybrid Platform for Geo-Temporal Enrichment of Digital Photo Collections Based on Computational and Crowdsourced Metadata Generation](https://link.springer.com/chapter/10.1007/978-3-030-01762-0_10)



