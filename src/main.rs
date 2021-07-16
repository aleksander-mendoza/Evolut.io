mod fps;
mod input;

extern crate nalgebra_glm as glm;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState, SubpassContents, CommandBufferUsage};
use vulkano::device::{Device, DeviceExtensions, Queue};
use vulkano::image::{ImageUsage, SwapchainImage, ImageViewAbstract};
use vulkano::instance::{Instance, PhysicalDevice, PhysicalDeviceType, QueueFamily};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::{swapchain, Version};
use vulkano::swapchain::{AcquireError, ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError, Surface, CompositeAlpha};
use vulkano::sync;
use vulkano::format::Format;
use vulkano::sync::{FlushError, GpuFuture, SharingMode};
use vulkano::device::Features;
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent, KeyboardInput, ElementState, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use std::sync::Arc;
use vulkano::render_pass::{FramebufferAbstract, Subpass, RenderPass, Framebuffer};
use vulkano::pipeline::vertex::{VertexSource, OneVertexOneInstanceDefinition};
use vulkano::image::view::ImageView;
use failure::err_msg;
use vulkano::impl_vertex;
use crate::fps::FpsCounter;
use winit::event::DeviceEvent::Key;
use crate::input::Input;

fn vulkan_instance() -> Result<Arc<Instance>, failure::Error> {
    let required_extensions = vulkano_win::required_extensions();

    // Now creating the instance.
    let instance = Instance::new(None, Version::V1_2, &required_extensions, None);

    instance.map_err(err_msg)
}

fn device(instance: &Arc<Instance>) -> Result<PhysicalDevice, failure::Error> {
    let mut devices = PhysicalDevice::enumerate(instance);
    let mut best = devices.next().ok_or_else(|| err_msg("No GPU devices available!"))?;
    let mut best_score = 0;
    for device in devices {
        let score = match device.properties().device_type {
            Some(PhysicalDeviceType::DiscreteGpu) => 100,
            Some(PhysicalDeviceType::IntegratedGpu) => 50,
            Some(PhysicalDeviceType::VirtualGpu) => 10,
            _ => 0
        };
        if score > best_score {
            best_score = score;
            best = device;
        }
    }
    Ok(best)
}

fn find_surface<'a>(device: &'a PhysicalDevice, surface: &Arc<Surface<Window>>) -> Result<QueueFamily<'a>, failure::Error> {
    device
        .queue_families()
        .find(|&q| {
            // We take the first queue that supports drawing to our window.
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        }).ok_or_else(|| err_msg("Could not find a suitable queue family"))
}

fn device_and_queue(physical: PhysicalDevice, queue_family: QueueFamily) -> Result<(Arc<Device>, Arc<Queue>), failure::Error> {
    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    println!("{:?}", physical.supported_features());
    let (device, mut queues) = Device::new(
        physical,
        &Features::none(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    ).map_err(err_msg)?;

    // Since we can request multiple queues, the `queues` variable is in fact an iterator. In this
    // example we use only one queue, so we just retrieve the first and only element of the
    // iterator and throw it away.
    let queue = queues.next().ok_or_else(|| err_msg("No queues found for the specified family"))?;
    Ok((device, queue))
}

fn swapchains(device: &Arc<Device>, physical: PhysicalDevice, surface: &Arc<Surface<Window>>) -> Result<(Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>), failure::Error> {
    let caps = surface.capabilities(physical).map_err(err_msg)?;
    let (format, color_space) = caps.supported_formats.iter().find(|(format, color_space)| *format == Format::B8G8R8A8Srgb && *color_space == ColorSpace::SrgbNonLinear).cloned().unwrap_or(caps.supported_formats[0]);

    let dimensions: [u32; 2] = surface.window().inner_size().into();
    let num_images = (caps.min_image_count + 1).min(caps.max_image_count.unwrap_or(u32::MAX));
    // Please take a look at the docs for the meaning of the parameters we didn't mention.
    Swapchain::start(
        device.clone(),
        surface.clone())
        .num_images(num_images)
        .format(format)
        .dimensions(dimensions)
        .layers(1)
        .usage(ImageUsage::color_attachment())
        .sharing_mode(SharingMode::Exclusive)
        .transform(SurfaceTransform::Identity)
        .composite_alpha(CompositeAlpha::Opaque)
        .present_mode(if caps.present_modes.mailbox { PresentMode::Mailbox } else { PresentMode::Fifo })
        .fullscreen_exclusive(FullscreenExclusive::Default)
        .clipped(true)
        .color_space(color_space)
        .build()
        .map_err(err_msg)
}


fn render_pass(device: &Arc<Device>, swapchain: &Arc<Swapchain<Window>>) -> Result<Arc<RenderPass>, failure::Error> {
    vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                // `color` is a custom name we give to the first and only attachment.
                color: {
                    // `load: Clear` means that we ask the GPU to clear the content of this
                    // attachment at the start of the drawing.
                    load: Clear,
                    // `store: Store` means that we ask the GPU to store the output of the draw
                    // in the actual image. We could also ask it to discard the result.
                    store: Store,
                    // `format: <ty>` indicates the type of the format of the image. This has to
                    // be one of the types of the `vulkano::format` module (or alternatively one
                    // of your structs that implements the `FormatDesc` trait). Here we use the
                    // same format as the swapchain.
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                // We use the attachment named `color` as the one and only color attachment.
                color: [color],
                // No depth-stencil attachment is indicated with empty brackets.
                depth_stencil: {}
            }
        ).map_err(err_msg).map(Arc::new)
}

fn create_pipeline(device: &Arc<Device>, render_pass: &Arc<RenderPass>) -> Result<Arc<GraphicsPipeline<OneVertexOneInstanceDefinition<Vertex, InstanceData>>>, failure::Error> {
    let vs = vs::Shader::load(device.clone()).map_err(err_msg)?;
    let fs = fs::Shader::load(device.clone()).map_err(err_msg)?;

    // Before we draw we have to create what is called a pipeline. This is similar to an OpenGL
    // program, but much more specific.
    GraphicsPipeline::start()
        // We need to indicate the layout of the vertices.
        // The type `SingleBufferDefinition` actually contains a template parameter corresponding
        // to the type of each vertex. But in this code it is automatically inferred.
        .vertex_input(OneVertexOneInstanceDefinition::<Vertex, InstanceData>::new())
        // A Vulkan shader can in theory contain multiple entry points, so we have to specify
        // which one. The `main` word of `main_entry_point` actually corresponds to the name of
        // the entry point.
        .vertex_shader(vs.main_entry_point(), ())
        // The content of the vertex buffer describes a list of triangles.
        .triangle_list()
        // Use a resizable viewport set to draw over the entire window
        .viewports_dynamic_scissors_irrelevant(1)
        // See `vertex_shader`.
        .fragment_shader(fs.main_entry_point(), ())
        // We have to indicate which subpass of which render pass this pipeline is going to be used
        // in. The pipeline will only be usable from this particular subpass.
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
        .build(device.clone())
        .map_err(err_msg).map(Arc::new)
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "assets/shaders/blocks.vert"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "assets/shaders/blocks.frag"
    }
}

// # Vertex Types
//
// Seeing as we are going to use the `OneVertexOneInstanceDefinition` vertex definition for our
// graphics pipeline, we need to define two vertex types:
//
// 1. `Vertex` is the vertex type that we will use to describe the triangle's geometry.
#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 2],
}
impl_vertex!(Vertex, position);

// 2. `InstanceData` is the vertex type that describes the unique data per instance.
#[derive(Default, Debug, Clone)]
struct InstanceData {
    position_offset: [f32; 2],
    scale: f32,
}
impl_vertex!(InstanceData, position_offset, scale);

fn main() -> Result<(), failure::Error> {
    let instance = vulkan_instance()?;
    let physical = device(&instance)?;
    println!("Using device: {:?}", physical.properties());
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .map_err(err_msg)?;
    let queue_family = find_surface(&physical, &surface)?;
    let (device, queue) = device_and_queue(physical, queue_family)?;
    let (mut swapchain, images) = swapchains(&device, physical, &surface)?;
    let render_pass = render_pass(&device, &swapchain)?;
    let pipeline = create_pipeline(&device, &render_pass)?;
    let mut fps_counter = FpsCounter::new(60);
    // We now create a buffer that will store the shape of our triangle.
    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        [
            Vertex {
                position: [-0.5, -0.25],
            },
            Vertex {
                position: [0.0, 0.5],
            },
            Vertex {
                position: [0.25, -0.1],
            },
        ].into_iter().cloned(),
    ).map_err(err_msg)?;
    // Now we create another buffer that will store the unique data per instance.
    // For this example, we'll have the instances form a 10x10 grid that slowly gets larger.
    let instance_data_buffer = {
        let rows = 10;
        let cols = 10;
        let n_instances = rows * cols;
        let mut data = Vec::new();
        for c in 0..cols {
            for r in 0..rows {
                let half_cell_w = 0.5 / cols as f32;
                let half_cell_h = 0.5 / rows as f32;
                let x = half_cell_w + (c as f32 / cols as f32) * 2.0 - 1.0;
                let y = half_cell_h + (r as f32 / rows as f32) * 2.0 - 1.0;
                let position_offset = [x, y];
                let scale = (2.0 / rows as f32) * (c * rows + r) as f32 / n_instances as f32;
                data.push(InstanceData {
                    position_offset,
                    scale,
                });
            }
        }
        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            data.iter().cloned(),
        ).map_err(err_msg)?
    };


    // Dynamic viewports allow us to recreate just the viewport when the window is resized
    // Otherwise we would have to recreate the whole pipeline.
    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: None,
        scissors: None,
        compare_mask: None,
        write_mask: None,
        reference: None,
    };

    // The render pass we created above only describes the layout of our framebuffers. Before we
    // can draw we also need to create the actual framebuffers.
    //
    // Since we need to draw to multiple images, we are going to create a different framebuffer for
    // each image.
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

    // Initialization is finally finished!

    // In some situations, the swapchain will become invalid by itself. This includes for example
    // when the window is resized (as the images of the swapchain will no longer match the
    // window's) or, on Android, when the application went to the background and goes back to the
    // foreground.
    //
    // In this situation, acquiring a swapchain image or presenting it will return an error.
    // Rendering to an image of that swapchain will not produce any error, but may or may not work.
    // To continue rendering, we need to recreate the swapchain by creating a new swapchain.
    // Here, we remember that we need to do this for the next loop iteration.
    let mut recreate_swapchain = false;

    // In the loop below we are going to submit commands to the GPU. Submitting a command produces
    // an object that implements the `GpuFuture` trait, which holds the resources for as long as
    // they are in use by the GPU.
    //
    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let mut position = glm::vec3(0f32, 0., 0.);
    let movement_speed = 0.001f32;
    let mut input = Input::new();
    // Specify the color to clear the framebuffer with i.e. blue

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput {
                    input: KeyboardInput { state, virtual_keycode:Some(key), .. }, ..
                }, ..
            } => {
                input.update_keyboard(state, key)
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved {
                    position, ..
                }, ..
            } => {
                input.update_mouse_position(position)
            }
            Event::WindowEvent {
                event: WindowEvent::MouseInput {
                     state, button, ..
                }, ..
            } => {
                input.update_mouse_click(state,button)
            }
            Event::MainEventsCleared => {
                fps_counter.update();
                let movement_vector = input.get_direction_unit_vector() * movement_speed * fps_counter.delta_f32();

                // Before we can draw on the output, we have to *acquire* an image from the swapchain. If
                // no image is available (which happens if you submit draw commands too quickly), then the
                // function will block.
                // This operation returns the index of the image that we are allowed to draw upon.
                //
                // This function can block if no image is available. The parameter is an optional timeout
                // after which the function call will return an error.
                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                // acquire_next_image can be successful, but suboptimal. This means that the swapchain image
                // will still work, but it may not display correctly. With some drivers this can be when
                // the window resizes, but it may not cause the swapchain to become out of date.
                if suboptimal {
                    recreate_swapchain = true;
                }
                // In order to draw, we have to build a *command buffer*. The command buffer object holds
                // the list of commands that are going to be executed.
                //
                // Building a command buffer is an expensive operation (usually a few hundred
                // microseconds), but it is known to be a hot path in the driver and is expected to be
                // optimized.
                //
                // Note that we have to pass a queue family when we create the command buffer. The command
                // buffer will only be executable on that given queue family.
                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                ).unwrap();
                let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];
                builder
                    // Before we can draw, we have to *enter a render pass*. There are two methods to do
                    // this: `draw_inline` and `draw_secondary`. The latter is a bit more advanced and is
                    // not covered here.
                    //
                    // The third parameter builds the list of values to clear the attachments with. The API
                    // is similar to the list of attachments when building the framebuffers, except that
                    // only the attachments that use `load: Clear` appear in the list.
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        clear_values,
                    )
                    .unwrap()
                    // We are now inside the first subpass of the render pass. We add a draw command.
                    //
                    // The last two parameters contain the list of resources to pass to the shaders.
                    // Since we used an `EmptyPipeline` object, the objects have to be `()`.
                    .draw(
                        pipeline.clone(),
                        &dynamic_state,
                        // We pass both our lists of vertices here.
                        (vertex_buffer.clone(), instance_data_buffer.clone()),
                        (),
                        (),
                        [],
                    )
                    .unwrap()
                    // We leave the render pass by calling `draw_end`. Note that if we had multiple
                    // subpasses we could have called `next_inline` (or `next_secondary`) to jump to the
                    // next subpass.
                    .end_render_pass()
                    .unwrap();

                // Finish building the command buffer by calling `build`.
                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    // The color output is now expected to contain our triangle. But in order to show it on
                    // the screen, we have to *present* the image by calling `present`.
                    //
                    // This function does not actually present the image immediately. Instead it submits a
                    // present command at the end of the queue. This means that it will only be presented once
                    // the GPU has finished executing the command buffer that draws the triangle.
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            Event::RedrawEventsCleared => {
                input.reset();
                // It is important to call this function from time to time, otherwise resources will keep
                // accumulating and you will eventually reach an out of memory error.
                // Calling this function polls various fences in order to determine what the GPU has
                // already processed, and frees the resources that are no longer needed.
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                // Whenever the window resizes we need to recreate everything dependent on the window size.
                // In this example that includes the swapchain, the framebuffers and the dynamic state viewport.
                if recreate_swapchain {
                    // Get the new dimensions of the window.
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) =
                        match swapchain.recreate().dimensions(dimensions).build() {
                            Ok(r) => r,
                            // This error tends to happen when the user is manually resizing the window.
                            // Simply restarting the loop is the easiest way to fix this issue.
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;
                    // Because framebuffers contains an Arc on the old swapchain, we need to
                    // recreate framebuffers as well.
                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut dynamic_state,
                    );
                    recreate_swapchain = false;
                }
            }
            _ => (),
        }
    });
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(ImageView::new(image.clone()).unwrap())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}