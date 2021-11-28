const BLOCK_SENSORY_FEATURES_LEN:usize = 6;
const ANN_LIDAR_COUNT:usize = 32;
const BLOCK_EXTENDED_SENSORY_FEATURES_LEN:usize = BLOCK_SENSORY_FEATURES_LEN+2;// + block_mass and is_air
const ANN_TOUCHED_BLOCK_COUNT:usize = 8;//cube has 8 corners
const ANN_INPUT_SIZE:usize = ANN_LIDAR_COUNT+BLOCK_EXTENDED_SENSORY_FEATURES_LEN*ANN_TOUCHED_BLOCK_COUNT;
const ANN_HIDDEN_SIZE:usize = 32;
const ANN_INPUT_CONNECTIONS_PER_HIDDEN_NEURON:usize = 16;
const ANN_LATENT_SIZE:usize = 32;
const ANN_HIDDEN_CONNECTIONS_PER_LATENT_NEURON:usize = 4;
const ANN_LATENT_CONNECTIONS_PER_LATENT_NEURON:usize = 4;
const ANN_OUTPUT_ATTACK_MUSCLES_SIZE:usize = 4;
const ANN_OUTPUT_MOVEMENT_MUSCLES_SIZE:usize = 20;
const ANN_OUTPUT_ROTATION_MUSCLES_SIZE:usize = 8;
const ANN_OUTPUT_SIZE:usize = ANN_OUTPUT_MOVEMENT_MUSCLES_SIZE+ANN_OUTPUT_ATTACK_MUSCLES_SIZE+ANN_OUTPUT_ROTATION_MUSCLES_SIZE;
const ANN_LATENT_CONNECTIONS_PER_OUTPUT_NEURON:usize = 4;
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C, packed)]
pub struct AnnSparseConnection{
    src_neuron:u32,
    weight:f32
}
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C, packed)]
pub struct AnnLidar{
    direction:glm::Vec3,
    dummy:f32,
}
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C, packed)]
pub struct AnnSparseHiddenNeuron{
    incoming: [AnnSparseConnection;ANN_INPUT_CONNECTIONS_PER_HIDDEN_NEURON],
    bias:f32,
}
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C, packed)]
pub struct AnnSparseLatentNeuron{
    incoming_from_hidden:[AnnSparseConnection;ANN_HIDDEN_CONNECTIONS_PER_LATENT_NEURON] ,
    recurrent_from_latent:[AnnSparseConnection;ANN_LATENT_CONNECTIONS_PER_LATENT_NEURON] ,
    bias:f32,
}
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C, packed)]
pub struct AnnSparseOutputNeuron{
    incoming:[AnnSparseConnection;ANN_LATENT_CONNECTIONS_PER_OUTPUT_NEURON],
    bias:f32,
}
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C, packed)]
pub struct AnnEntity{
    ann_hidden:[AnnSparseHiddenNeuron;ANN_HIDDEN_SIZE] ,
    ann_latent:[AnnSparseLatentNeuron;ANN_LATENT_SIZE] ,
    ann_output:[AnnSparseOutputNeuron;ANN_OUTPUT_SIZE] ,
    latent:[f32;ANN_LATENT_SIZE] ,
    lidars:[AnnLidar;ANN_LIDAR_COUNT] ,
    bone_idx:u32,
    main:u32,
    energy:f32,
    speed:f32,
}
