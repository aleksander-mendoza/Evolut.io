#[derive(Copy, Clone, Debug)]
#[repr(C,packed)]
pub struct NeuralNetLayerInput{
    region0_offset:u32,
    region1_offset:u32,
    region2_offset:u32,
    region0_length:u32,
    region1_length:u32,
    region2_length:u32,
}
#[derive(Copy, Clone, Debug)]
#[repr(C,packed)]
pub struct NeuralNetLayerHidden{
    input_offset:u32,
    input_length:u32,
    weights_offset:u32,
    output_offset:u32,
    output_length:u32,
    next_layer:u32,
}
#[derive(Copy, Clone, Debug)]
#[repr(C,packed)]
pub struct NeuralNetLayerOutput{
    muscle_internal_offset:u32,
    muscle_external_offset:u32,
    muscle_length:u32,
    recurrent_internal_offset:u32,
    recurrent_external_offset:u32,
    recurrent_length:u32,
}
#[derive(Copy, Clone)]
#[repr(C,packed)]
pub union NeuralNetLayer{
    pub input: NeuralNetLayerInput,
    pub hidden: NeuralNetLayerHidden,
    pub output: NeuralNetLayerOutput,
}
pub enum Aggregate{
    Overwrite,
    Sum,
}
impl NeuralNetLayer{
    pub fn new_output(muscle_internal_offset:u32,
                      muscle_external_offset:u32,
                      muscle_length:u32,
                      recurrent_internal_offset:u32,
                      recurrent_external_offset:u32,
                      recurrent_length:u32,)->Self{
        NeuralNetLayer{output:NeuralNetLayerOutput{
            muscle_internal_offset,
            muscle_external_offset,
            muscle_length,
            recurrent_internal_offset,
            recurrent_external_offset,
            recurrent_length
        }}
    }
    pub fn new_hidden(input_offset:u32,input_length:u32,weights_offset:u32,output_offset:u32,output_length:u32,next_layer:Option<u32>, aggregate:Aggregate)->Self{
        NeuralNetLayer{hidden:NeuralNetLayerHidden{
            input_offset,
            input_length,
            weights_offset,
            output_offset,
            output_length,
            next_layer:2*next_layer.unwrap_or(u16::MAX as u32 + 1)+match aggregate{
                Aggregate::Overwrite => 1,
                Aggregate::Sum => 0
            }
        }}
    }

    pub fn new_input(offset:u32,length:u32)->Self{
        Self::new_input_recurrent(offset,length,0,0)
    }

    pub fn new_input_recurrent(sensor_offset:u32,sensor_length:u32,recurrent_offset:u32,recurrent_length:u32)->Self{
        NeuralNetLayer{input:NeuralNetLayerInput{
            region0_offset: sensor_offset,
            region1_offset: recurrent_offset,
            region2_offset: 0,
            region0_length: sensor_length,
            region1_length: recurrent_length,
            region2_length: 0
        }}
    }
}
