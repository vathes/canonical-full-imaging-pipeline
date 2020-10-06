from my_project import subject, imaging, lab, schema

imaging.schema.drop()
schema.drop()
subject.schema.drop()
lab.schema.drop()

# import datajoint as dj
# dj.Diagram(subject)