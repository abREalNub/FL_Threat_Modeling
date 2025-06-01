
def muestra():
    print("Estoy entrando")


def count_neurons_for_each_layer(params):#input_height, input_width , stride = 1, padding = 1
    dimensions_of_layers = {}
    neurons = {}
    count_layer = 0

    for name, param in params:
        total_neuron_in_this = 0
        dimensions = param.size() #Sino shape, sino convertir a numnpy y sacar el shape
        print("Dimensiones del param:", len(dimensions),"Nombre", name)
        
        if "weight" in name and len(dimensions)>1:
            if "fc" in name:
                in_features = param.size(1)
                out_features = param.size(0)
                total_neuron_in_this = out_features #tratar solo con in no con aut, porque out son las de sealida, por ejemplo en la última capa no hay salida
            elif "conv" in name:
                print(name, param.size(0))
                #in_features = param.size(1)# Ver esto igualmente
                out_features = param.size(0)
                #kernel_size = param.size(2)
                
                #output_height = (input_height - kernel_size + 2 * padding) // stride + 1 # Se asume mucho acá
                #output_width = (input_width - kernel_size + 2 * padding) // stride + 1   # Se asume mucho acá

                #input_height, input_width = output_height, output_width

                total_neuron_in_this = out_features 
            elif "out" in name:
                out_features = param.size(0)
                total_neuron_in_this = out_features
            name_dict = name + str(count_layer)
            count_layer += 1
            neurons[name_dict] = total_neuron_in_this
            dimensions_of_layers [name_dict] = dimensions
        #elif "bias" in name:
        #    out_features = param.size(0)
        #    total_neuron_in_this = out_features

        #neurons[name_dict] = total_neuron_in_this
        #dimensions_of_layers [name_dict] = dimensions




    return dimensions_of_layers, neurons

def extract_weigth_and_bias(params_dict):
    bias = []
    weight = []
    print(len(params_dict))
    for named, value in params_dict:
        print(named)
        if "bias" in named or 2 > len(value.size()): 
            bias.append(value)
        elif "weight" in named:
            weight.append(value)
    print(len(weight), len(bias))
    return weight, bias