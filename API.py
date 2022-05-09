from pycaret.regression import *


loaded_model = load_model('my_best_model')

print(loaded_model)

# create api
create_api(loaded_model, 'rf_model_api')
