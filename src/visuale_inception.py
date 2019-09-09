import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

model = models.InceptionV1()
model.load_graphdef()

channel = lambda n: objectives.channel("mixed4a_pre_relu", n)
obj = channel(122) + channel(155)
_ = render.render_vis(model, obj)
