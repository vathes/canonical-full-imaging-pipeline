import datajoint as dj
from djsubject import subject
from djlab import lab
from djimaging import imaging
from my_project import db_prefix

# ============== Declare "lab" and "subject" schema ==============

lab.declare(db_prefix + 'lab')

subject.declare(db_prefix + 'subject',
                dependencies={'Source': lab.Source,
                              'Lab': lab.Lab,
                              'Protocol': lab.Protocol,
                              'User': lab.User})


# ---- add "Scanner" table ----

@lab.schema
class Scanner(dj.Manual):
    definition = """
    scanner: varchar(32)    
    """


# ============== Declare Session table ==============

schema = dj.schema(db_prefix + 'experiment')


@schema
class Session(dj.Manual):
    definition = """
    -> subject.Subject
    session_datetime: datetime
    """


# ============== Declare "imaging" schema ==============

imaging.declare(dj.schema(db_prefix + 'imaging'),
                dependencies={'Session': Session,
                              'Equipment': Scanner,
                              'Location': lab.Location,
                              'get_scan_image_files': lambda **x: "Not yet implemented",
                              'get_suite2p_dir': lambda **x: "Not yet implemented"})

