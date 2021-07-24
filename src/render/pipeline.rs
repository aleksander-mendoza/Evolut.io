use crate::render::shader_module::ShaderModule;
use ash::vk;
use crate::render::device::Device;
use ash::version::DeviceV1_0;
use std::ffi::{CStr, CString};
use failure::err_msg;
use crate::render::render_pass::{RenderPassBuilder, RenderPass};
use std::rc::Rc;
use ash::vk::PipelineLayout;
use crate::render::data::VertexSource;
use crate::render::descriptor_layout::DescriptorLayout;
use crate::render::buffer::{Buffer, Type};


pub struct Pipeline {
    raw: vk::Pipeline,
    layout: vk::PipelineLayout,
    render_pass: RenderPass, // Keeping this reference prevents render pass from being deallocated
    // before pipeline. While the specification says that it's not necessary and in principle RenderPass
    // could outlive pipeline, some vendors may have bugs in their implementations. It's a lot
    // safer to keep this reference just in case.
    descriptor_layout: Vec<DescriptorLayout>, // Just keeping reference
}

impl Pipeline {
    pub fn device(&self) -> &Device {
        self.render_pass.device()
    }
    pub fn raw(&self) -> vk::Pipeline {
        self.raw
    }
    pub fn layout(&self) -> vk::PipelineLayout{
        self.layout
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device().inner().destroy_pipeline(self.raw, None);
            self.device().inner().destroy_pipeline_layout(self.layout, None);
            // Safety: The pipeline is dropped first. Then the layout and the render pass is dropped
            // last (unless some other pipeline uses it too)
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
    topology: vk::PrimitiveTopology,
    vertex_input_attribute:Vec<vk::VertexInputAttributeDescription>,
    vertex_input_binding:Vec<vk::VertexInputBindingDescription>,
    descriptor_layout: Vec<DescriptorLayout>

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
            rasterizer: vk::PipelineRasterizationStateCreateInfo::builder()
                .line_width(1.0)
                .build(),
            multisample_state_create_info: vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                .build(),
            depth_state_create_info: vk::PipelineDepthStencilStateCreateInfo::builder()
                .front(stencil_state)
                .back(stencil_state)
                .min_depth_bounds(0.0)
                .max_depth_bounds(1.0)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false)
                .depth_compare_op(vk::CompareOp::LESS)
                .build(),
            color_blend_attachment_states: vec![],
            color_blend_state: vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op(vk::LogicOp::COPY)
                .blend_constants([0.0, 0.0, 0.0, 0.0])
                .build(),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            vertex_input_attribute: vec![],
            vertex_input_binding: vec![],
            descriptor_layout: vec![]
        }
    }
    pub fn depth_bounds(mut self, min:f32,max:f32)->Self{
        self.depth_state_create_info.depth_bounds_test_enable = vk::TRUE;
        self.depth_state_create_info.max_depth_bounds = max;
        self.depth_state_create_info.min_depth_bounds = min;
        self
    }
    pub fn stencil_test(mut self, enable:bool)->Self{
        self.depth_state_create_info.stencil_test_enable = enable.into();
        self
    }
    pub fn depth_test(mut self, enable:bool)->Self{
        self.depth_state_create_info.depth_test_enable = enable.into();
        self.depth_state_create_info.depth_write_enable = enable.into();
        self
    }
    pub fn descriptor_layout(mut self, layout:DescriptorLayout)->Self{
        self.descriptor_layout.push(layout);
        self
    }

    pub fn color_blend_attachment_states(mut self, blend_state: vk::PipelineColorBlendAttachmentState) -> Self {
        self.color_blend_attachment_states.push(blend_state);
        self
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

    pub fn topology(mut self,topology:vk::PrimitiveTopology)->Self{
        self.topology = topology;
        self
    }
    pub fn vertex_input<V:VertexSource>(self, binding:u32, _buffer:&Buffer<V,impl Type>)->Self{
        self.input_buffer(binding,_buffer,vk::VertexInputRate::VERTEX)
    }
    pub fn instance_input<V:VertexSource>(self, binding:u32, _buffer:&Buffer<V,impl Type>)->Self{
        self.input_buffer(binding,_buffer,vk::VertexInputRate::INSTANCE)
    }
    pub fn input_buffer<V:VertexSource>(mut self, binding:u32, _buffer:&Buffer<V,impl Type>, input_rate:vk::VertexInputRate)->Self{
        self.vertex_input_binding.push(vk::VertexInputBindingDescription {
            binding,
            stride: std::mem::size_of::<V>() as u32,
            input_rate,
        });
        for attr in V::get_attribute_descriptions(binding){
            self.vertex_input_attribute.push(attr);
        }
        self
    }

    pub fn build(&mut self, render_pass: &RenderPass) -> Result<Pipeline, failure::Error> {
        let Self {
            viewport,
            scissors,
            shaders,
            rasterizer,
            multisample_state_create_info,
            depth_state_create_info,
            color_blend_attachment_states,
            color_blend_state,
            topology,
            vertex_input_attribute,
            vertex_input_binding,
            descriptor_layout,
        } = self;
        let vp = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(viewport)
            .scissors(scissors);
        let shader_names: Vec<CString> = shaders.iter().map(|(name, _)| CString::new(name.as_bytes()).expect("Name of shader's main function contains illegal null \\0 symbol")).collect();
        let shader_stages: Vec<vk::PipelineShaderStageCreateInfo> = shader_names.iter().zip(shaders).map(|(c_name, (_, shader))| shader.to_stage_info(c_name).build()).collect();
        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vertex_input_attribute)
            .vertex_binding_descriptions(&vertex_input_binding);
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo::builder().topology(*topology);
        color_blend_state.attachment_count = color_blend_attachment_states.len() as u32;
        color_blend_state.p_attachments = color_blend_attachment_states.as_ptr();
        let set_layouts:Vec<vk::DescriptorSetLayout> = descriptor_layout.iter().map(|s|s.raw()).collect();
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts);
        let pipeline_layout = unsafe { render_pass.device().inner()
            .create_pipeline_layout(&pipeline_layout_create_info, None) }?;
        let p = vk::GraphicsPipelineCreateInfo::builder()
            .viewport_state(&vp)
            .stages(shader_stages.as_slice())
            .rasterization_state(rasterizer)
            .vertex_input_state(&vertex_input_state_create_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .color_blend_state(color_blend_state)
            .depth_stencil_state(depth_state_create_info)
            .multisample_state(multisample_state_create_info)
            .layout(pipeline_layout)
            .base_pipeline_index(-1)
            .render_pass(render_pass.raw())
            .subpass(0);
        let result = unsafe {
            render_pass.device().inner().create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&p),
                None,
            )
        };
        fn new(pipeline: Vec<vk::Pipeline>, pipeline_layout: PipelineLayout, render_pass: &RenderPass, descriptor_layout:&Vec<DescriptorLayout>) -> Pipeline {
            Pipeline {
                raw: pipeline.into_iter().next().unwrap(),
                layout: pipeline_layout,
                render_pass: render_pass.clone(),
                descriptor_layout: descriptor_layout.clone()
            }
        }
        match result {
            Ok(pipeline) => Ok(new(pipeline, pipeline_layout, render_pass, descriptor_layout)),
            Err((pipeline, err)) => {
                new(pipeline, pipeline_layout, render_pass, descriptor_layout);
                Err(err_msg(err))
            }
        }
    }
}