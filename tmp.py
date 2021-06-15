

import models

model_names = sorted(name for name in models.__dict__
                     if not name.startwith("__")
                     and callable (models.__dict__[name])
                     )
