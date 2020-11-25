import datajoint as dj
from djsubject import subject   # canonical-colony-management
from djlab import lab           # canonical-lab-management
from djimaging import imaging   # canonical-imaging

from utils.path_utils import get_imaging_root_data_dir, get_scan_image_files, get_suite2p_dir


if 'custom' not in dj.config:
    dj.config['custom'] = {}

db_prefix = dj.config['custom'].get('database.prefix', '')


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
                              'get_root_data_dir': get_imaging_root_data_dir,
                              'get_scan_image_files': get_scan_image_files,
                              'get_suite2p_dir': get_suite2p_dir})
