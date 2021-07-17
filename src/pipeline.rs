use crate::shader_module::ShaderModule;
use ash::vk;
use crate::device::Device;
use ash::version::DeviceV1_0;
use ash::vk::{Viewport, Rect2D, PolygonMode};
use std::ffi::{CStr, CString};

pub struct PipelineBuilder {
    viewport: Vec<Viewport>,
    scissors: Vec<Rect2D>,
    shaders:Vec<(String,ShaderModule)>,
    rasterizer: vk::PipelineRasterizationStateCreateInfo
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            viewport: vec![],
            scissors: vec![],
            shaders: vec![],
            rasterizer: vk::PipelineRasterizationStateCreateInfo::builder().build()
        }
    }

    pub fn viewports(mut self, viewport: Viewport) -> Self {
        self.viewport.push(viewport);
        self
    }

    pub fn scissors(mut self, scissors: Rect2D) -> Self {
        self.scissors.push(scissors);
        self
    }

    pub fn cull_face(mut self, cull_mode:vk::CullModeFlags) -> Self {
        self.rasterizer.cull_mode = cull_mode;
        self
    }

    pub fn front_face_clockwise(mut self, clockwise:bool) -> Self {
        self.rasterizer.front_face = if clockwise{vk::FrontFace::CLOCKWISE}else{vk::FrontFace::COUNTER_CLOCKWISE};
        self
    }

    pub fn line_width(mut self, line_width:f32) -> Self {
        self.rasterizer.line_width = line_width;
        self
    }

    pub fn polygon_mode(mut self, polygon_mode:PolygonMode) -> Self {
        self.rasterizer.polygon_mode = polygon_mode;
        self
    }

    pub fn shader(mut self, main_func: impl ToString, shader:ShaderModule) -> Self {
        self.shaders.push((main_func.to_string(),shader.clone()));
        self
    }

    pub fn build(self, device: &Device) -> Result<Vec<ash::vk::Pipeline>, (Vec<ash::vk::Pipeline>, ash::vk::Result)> {
        let Self { viewport, scissors, shaders, rasterizer } = self;
        let vp = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewport)
            .scissors(&scissors);
        let shader_names:Vec<CString> = shaders.iter().map(|(name,_)|CString::new(name.as_bytes()).expect("Name of shader's main function contains illegal null \\0 symbol")).collect();
        let shader_stages:Vec<vk::PipelineShaderStageCreateInfo> = shader_names.iter().zip(shaders.iter()).map(|(c_name,(_,shader))|shader.to_stage_info(c_name).build()).collect();
        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder();
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo::builder().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let p = vk::GraphicsPipelineCreateInfo::builder()
            .viewport_state(&vp)
            .stages(shader_stages.as_slice())
            .rasterization_state(&rasterizer)
            .vertex_input_state(&vertex_input_state_create_info)
            .input_assembly_state(&vertex_input_assembly_state_info);
        unsafe {
            device.device().create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&p),
                None,
            )
        }
    }
}