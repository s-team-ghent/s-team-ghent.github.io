import os
import yaml
import shutil

yaml_settings = yaml.safe_load(open(os.path.join('.', '_data','settings.yml'), 'r'))
authors_yaml = yaml.safe_load(open(os.path.join('.', '_data','authors.yml'), 'r'))

print("Creating a new project page on the website.")
print("-------------------------------------------")
print("Current projects are: ")
projects = yaml_settings['projects']
for project in projects:
    print("\t-", project['name'])
print("Enter project name: ")
project_name = input().upper()
print("Enter main researcher initials: ")
for k in authors_yaml:
    print("\t- {}".format(k))
research_lead = input()
print("Add other researchers, separated by ', ': ")
for k in authors_yaml:
    print("\t- {}".format(k))
other_researchers = input()
print("Creating project : ", project_name)
print("-Adding to settings.yml")
project_setting = {
    'name': project_name, 
    'folder': project_name, 
    'file': "projects/{}.html".format(project_name), 
    'lead_researcher': research_lead,
    'other_researchers': other_researchers.split(", ")
  }
yaml_settings["projects"].append(project_setting)
with open(os.path.join(".","_data","settings.yml"), 'w') as file:
    documents = yaml.dump(yaml_settings, file)
print("DONE")
print("-Creating projects/{}.md".format(project_name))
with open(os.path.join('.','projects',"{}.md".format(project_name)), 'w') as ofile:
    ofile.write('''---
layout: project
title: '{}'
---

## About the project

WIP

## IDLab role

IDLab has the following tasks within the {} project

WIP
'''.format(project_name, project_name))
print("DONE")
print("- Adding a thumbnail image for the project.")
print("  Please enter a valid path to a JPG image")
path_to_image = input()
# Create target Directory if don't exist
if not os.path.exists(os.path.join(".","assets","img",project_name)):
    os.mkdir(os.path.join(".","assets","img","projects", project_name))
    print("\t  Directory " , os.path.join(".","assets","img","projects",project_name) ,  " Created ")
else:    
    print("\t  Directory " , os.path.join(".","assets","img","projects", project_name) ,  " already exists")
shutil.copy(path_to_image, os.path.join(".","assets","img","projects",project_name,"thumb.jpg"))
print("DONE")