---
layout: project
title: 'BAS-X'
---

## About the project

Project in collaboration with [RouteYou](https://www.routeyou.com/) with the goal to analyze and improve the quality of their walking & cycling routes, while also developing tools to automatically extract new routes from raster maps. Through statistical analysis of real user data and correlation with other datasets (e.g. weather, road type) the route information can be updated to more accurately reflect the real world conditions. For instance, the platform can suggest detours when it has been raining recently, to avoid more dangerous and uncomfortable sections in the route.

We also developed tools to automatically geolocate raster maps and predict a gps track of the shown route. By recognizing the text on each map and using geocoders, an approximate geolocation for the map can be found. An image segmentation model was trained to automatically extract a mask of the route on the map. After postprocessing the extracted route, it can be georeferenced using the predicted location of the map, resulting in a gps track of the route. After matching the predicted route to the underlying road network, it can be automatically imported into Routeyou's database. Combined with a web scraper, this allows for automated discovery of new regions of interest and walking/cycling routes.  


## IDLab role

IDLab has the following tasks within BAS-X

1. Improving route metadata based on user generated data
2. Correlation study between route popularity and the route characteristics
3. Automated geolocation of walking and cycling maps
4. Extraction and georeferencing of the depicted routes

