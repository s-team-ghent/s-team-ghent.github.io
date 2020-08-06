---
layout: post
title: "Course penalty score"
tags: [data-science, gps]
author: jelledb
---
In this post we try to provide an automated penalty model to classify the last 3kms of cycling road races. (Text in Dutch)

#### Objectieve beoordeling van de &quot;ultimi chilometri&quot; in het wielrennen

_Wordt het – na alweer enkele bedenkelijke sprints in de voorbije dagen - niet eens tijd voor een uniform scoring mechanisme om de veiligheid van de laatste kilometers van wielerwedstrijden te beoordelen?_

Na de horrorcrash van Fabio Jakobsen in de eerste etappe van de Ronde van Polen ligt niet alleen Dylan Groenewegen onder vuur, maar krijgt ook de organisatie kritiek op de inplanting van de laatste kilometers van hun eerste etappe. Vaak wordt de sprint voorbereiding in de laatste kilometers extra geaccidenteerd door bijvoorbeeld gevaarlijke bochten of een dalende lijn die de snelheid van de treintjes nog verder de hoogte injaagt . Zijn dit geen zaken die we eigenlijk objectief zouden kunnen beoordelen? Met deze onderzoeksvraag gingen Steven Verstockt en Jelle De Bock van IDLab (een imec onderzoeksgroep aan Universiteit Gent) aan de slag. Een eerste objectieve metriek, gebaseerd op karakteristieken van de wegsegmenten van de laatste 3km, toont alvast aan dat het veiliger sprinten is in Kuurne Brussel Kuurne dan bijvoorbeeld in Milaan-Turijn (waar de sprint(voorbereiding) gisteren ook ontsierd werd door een valpartij).

Op basis van de route coördinaten van de laatste 3 kilometer en de informatie die beschikbaar is in platformen zoals OpenStreetMap (OSM) kan automatisch bepaald worden waar en hoeveel gevaarlijke punten (bochten, ronde punten, dalende segmenten, wegversmallingen, verkeersremmers, …) er zijn. Door deze te gaan wegen met de resterende afstand tot de finish en hun relatieve afstand ten opzichte van elkaar krijgen we een vrij uniforme score die kan helpen bij de keuze van het parcours, alsook bij de beoordeling ervan. Indien de informatie van OSM verouderd zou zijn of te weinig detail zou bevatten, kunnen wedstrijdjury of organisatoren deze metadata over de wegsegmenten ook zelf gaan annoteren - voor de laatste kilometers zou dit in principe vrij snel kunnen worden uitgevoerd. Men zou het zelfs ook kunnen eisen van de organisator om zulks een gedetailleerde beschrijving aan te leveren – het gaat hier immers over de veiligheid van de renners.

Een meer gedetailleerde studie, bijvoorbeeld op basis van videobeelden van de finale kilometers, behoort tevens ook tot de mogelijkheden en zou voor de aanvang van elke wedstrijd kunnen worden uitgevoerd (ter plaatse of op basis van StreetView beelden). Met behulp van machine learning kunnen deze beelden dan worden geanalyseerd en kan bijkomende informatie omtrent de toestand van het wegdek worden vergaard.

Voorlopige resultaten objectieve scoring:

|| Ronde Van Vlaanderen | Kuurne-Brussel-Kuurne | Ronde van Polen (etappe 1) | Milaan-Turijn |
|---| :---: |:---: | :---: | :---: |
| Turns Penalty| 9.4 | 2.3 | 12.8  | 11.7 |
| Roundabouts penaltly | 0| 0 | 21.5 | 21.4 |
| Speed penaltly |0 | 0  | 10.3 | 21.9 |
| Height penaltly | 4.0| 4.1 | 7.8 | 3.1|
| RoadType penaltly |0 | 0 | 6.5 | 21.6 |
| **Total Score** | **13.4** | **6.4** | **58.9** | **79.6** |

<br/>

Dit onderzoek kadert in het ICON DAIQUIRI project (Data &amp; Artificial Intelligence for QUantifIed ReportIng In sport)


**Contactpersoon:** prof. Steven Verstockt; [steven.verstockt@ugent.be](mailto:steven.verstockt@ugent.be); +32 474 65 52 41
