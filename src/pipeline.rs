use crate::shader_module::ShaderModule;
use ash::vk;
use crate::device::Device;
use ash::version::DeviceV1_0;
use std::ffi::{CStr, CString};
use failure::err_msg;

pub struct Pipeline{
    p: vk::Pipeline,
    layout: vk::PipelineLayout,
    device:Device
}

impl Drop for Pipeline{
    fn drop(&mut self) {
        unsafe {
            self.device.device().destroy_pipeline(self.p, None);
            self.device.device().destroy_pipeline_layout(self.layout, None);
        }
    }
}

pub struct PipelineBuilder {
    viewport: Vec<vk::Viewport>,
    scissors: Vec<vk::Rect2D>,
    shaders: Vec<(String, ShaderModule)>,
    rasterizer: vk::PipelineRasterizationStateCreateInfo,
    multisample_state_create_info: vk::PipelineMultisampleStateCreateInfo,
    depth_state_create_info: vk::PipelineDepthStencilStateCreateInfo,
    color_blend_attachment_states: Vec<vk::PipelineColorBlendAttachmentState>,
    color_blend_state: vk::PipelineColorBlendStateCreateInfo,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        let stencil_state = vk::StencilOpState::builder()
            .fail_op(vk::StencilOp::KEEP)
            .pass_op(vk::StencilOp::KEEP)
            .depth_fail_op(vk::StencilOp::KEEP)
            .compare_op(vk::CompareOp::ALWAYS)
            .build();
        Self {
            viewport: vec![],
            scissors: vec![],
            shaders: vec![],
            rasterizer: vk::PipelineRasterizationStateCreateInfo::builder().build(),
            multisample_state_create_info: vk::PipelineMultisampleStateCreateInfo::builder()
                .build(),
            depth_state_create_info: vk::PipelineDepthStencilStateCreateInfo::builder()
                .front(stencil_state)
                .back(stencil_state)
                .max_depth_bounds(1.0)
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                .build(),
            color_blend_attachment_states: vec![],
            color_blend_state: vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op(vk::LogicOp::COPY)
                .blend_constants([0.0, 0.0, 0.0, 0.0])
                .build(),
        }
    }

    pub fn viewports(mut self, viewport: vk::Viewport) -> Self {
        self.viewport.push(viewport);
        self
    }

    pub fn scissors(mut self, scissors: vk::Rect2D) -> Self {
        self.scissors.push(scissors);
        self
    }

    pub fn cull_face(mut self, cull_mode: vk::CullModeFlags) -> Self {
        self.rasterizer.cull_mode = cull_mode;
        self
    }

    pub fn front_face_clockwise(mut self, clockwise: bool) -> Self {
        self.rasterizer.front_face = if clockwise { vk::FrontFace::CLOCKWISE } else { vk::FrontFace::COUNTER_CLOCKWISE };
        self
    }

    pub fn line_width(mut self, line_width: f32) -> Self {
        self.rasterizer.line_width = line_width;
        self
    }

    pub fn polygon_mode(mut self, polygon_mode: vk::PolygonMode) -> Self {
        self.rasterizer.polygon_mode = polygon_mode;
        self
    }

    pub fn shader(mut self, main_func: impl ToString, shader: ShaderModule) -> Self {
        self.shaders.push((main_func.to_string(), shader.clone()));
        self
    }

    pub fn build(&mut self, device: &Device) -> Result<Pipeline, failure::Error> {
        let Self {
            viewport,
            scissors,
            shaders,
            rasterizer,
            multisample_state_create_info,
            depth_state_create_info,
            color_blend_attachment_states,
            color_blend_state
        } = self;
        let vp = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(viewport)
            .scissors(scissors);
        let shader_names: Vec<CString> = shaders.iter().map(|(name, _)| CString::new(name.as_bytes()).expect("Name of shader's main function contains illegal null \\0 symbol")).collect();
        let shader_stages: Vec<vk::PipelineShaderStageCreateInfo> = shader_names.iter().zip(shaders).map(|(c_name, (_, shader))| shader.to_stage_info(c_name).build()).collect();
        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder();
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo::builder().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        color_blend_state.attachment_count = color_blend_attachment_states.len() as u32;
        color_blend_state.p_attachments = color_blend_attachment_states.as_ptr();
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder();
        let pipeline_layout = unsafe {device.device().create_pipeline_layout(&pipeline_layout_create_info, None)}?;
        let p = vk::GraphicsPipelineCreateInfo::builder()
            .viewport_state(&vp)
            .stages(shader_stages.as_slice())
            .rasterization_state(rasterizer)
            .vertex_input_state(&vertex_input_state_create_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .color_blend_state(color_blend_state)
            .depth_stencil_state(depth_state_create_info)
            .multisample_state(multisample_state_create_info)
            .layout(pipeline_layout);
        let result = unsafe {
            device.device().create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&p),
                None,
            )
        };
        match result{
            Ok(pipeline) => Ok(Pipeline{
                p: pipeline.into_iter().next().unwrap(),
                layout: pipeline_layout,
                device: device.clone()
            }),
            Err((pipeline,err)) => {
                Pipeline{
                    p: pipeline.into_iter().next().unwrap(),
                    layout: pipeline_layout,
                    device: device.clone()
                };
                Err(err_msg(err))
            }
        }
    }
}