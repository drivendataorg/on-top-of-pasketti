"""``lele.layers`` placeholder — only ``Squeezeformer`` was used and the
final 11-model ensemble does not include the squeezeformer encoder, so
this stays empty. If you fork the repo to revive that backbone, vendor
``lele.layers.Squeezeformer`` here.
"""


def __getattr__(name):
  raise ImportError(
      f'lele.layers.{name} is not bundled in the standalone solution. '
      'The final 11-model ensemble does not use this layer; remove the '
      'import or vendor the original implementation if you need it.'
  )
