use image::{GenericImageView, EncodableLayout};
use crate::render::owned_buffer::{OwnedBuffer};
use crate::render::device::Device;
use ash::vk;
use std::marker::PhantomData;
use ash::vk::{Extent3D, DeviceMemory, Image, Extent2D};
use failure::err_msg;
use std::path::Path;
use crate::render::command_pool::{CommandPool};
use crate::render::imageview::{ImageView, Aspect, Color, Depth};

use crate::render::swap_chain::SwapChain;
use crate::render::submitter::Submitter;
use crate::render::buffer_type::Cpu;

pub trait Dim {
    const DIM: vk::ImageType;
    fn as_extent(&self) -> vk::Extent3D;
}

pub struct Dim2D {
    extent: vk::Extent3D,
}

impl Dim2D {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            }
        }
    }
}

impl From<Extent2D> for Dim2D {
    fn from(e: Extent2D) -> Self {
        Dim2D::new(e.width, e.height)
    }
}

impl Dim for Dim2D {
    const DIM: vk::ImageType = vk::ImageType::TYPE_2D;

    fn as_extent(&self) -> Extent3D {
        self.extent
    }
}

pub struct Texture<D: Dim, A:Aspect> {
    texture_image: Image,
    texture_image_memory: DeviceMemory,
    extent: vk::Extent3D,
    format: vk::Format,
    device: Device,
    _d: PhantomData<D>,
    _a:PhantomData<A>,
}

impl<D: Dim, A:Aspect> Drop for Texture<D, A> {
    fn drop(&mut self) {
        unsafe {
            self.device.inner().destroy_image(self.texture_image, None);
            self.device.inner().free_memory(self.texture_image_memory, None);
        }
    }
}

impl<D: Dim, A:Aspect> Texture<D, A> {
    pub fn format(&self) -> vk::Format {
        self.format
    }
    // pub fn mem_capacity(&self) -> DeviceSize {
    //     unsafe { self.device.inner().get_image_memory_requirements(self.raw()) }.size
    // }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn raw(&self) -> Image {
        self.texture_image
    }
    pub fn extent(&self) -> vk::Extent3D {
        self.extent
    }
    pub fn create_view(&self) -> Result<ImageView<A>, ash::vk::Result> {
        ImageView::new(self.raw(), self.format(), self.device())
    }
    pub fn new(device: &Device, format: vk::Format, dim: D) -> Result<Self, vk::Result> {
        let layout = vk::ImageLayout::UNDEFINED;
        let extent = dim.as_extent();
        let image_create_info = vk::ImageCreateInfo::builder()
            .image_type(D::DIM)
            .extent(extent)
            .format(format)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(A::USAGE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(layout);

        let texture_image = unsafe { device.inner().create_image(&image_create_info, None) }?;

        let image_memory_requirement = unsafe { device.inner().get_image_memory_requirements(texture_image) };
        let memory_type = device.find_memory_type(image_memory_requirement, vk::MemoryPropertyFlags::DEVICE_LOCAL);
        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(image_memory_requirement.size)
            .memory_type_index(memory_type);

        let texture_image_memory = unsafe { device.inner().allocate_memory(&memory_allocate_info, None) }?;
        unsafe {
            device.inner().bind_image_memory(texture_image, texture_image_memory, 0)?
        }
        Ok(Self { texture_image, texture_image_memory, format, device: device.clone(), extent, _d: PhantomData , _a: PhantomData })
    }
}


pub struct TextureView<D: Dim, A:Aspect> {
    texture: Texture<D, A>,
    imageview: ImageView<A>,
}

impl<D: Dim, A:Aspect> TextureView<D, A> {
    pub fn format(&self) -> vk::Format {
        self.texture().format()
    }
    pub fn imageview(&self) -> &ImageView<A> {
        &self.imageview
    }
    pub fn texture(&self) -> &Texture<D, A> {
        &self.texture
    }
    pub fn new(device: &Device, format: vk::Format, dim: D) -> Result<Self, failure::Error> {
        let texture = Texture::new(device, format, dim)?;
        let imageview = texture.create_view()?;
        Ok(Self { texture, imageview })
    }
}

impl <D:Dim> TextureView<D,Color>{
    pub fn empty_image(device: &Device, format: vk::Format, dim: D) -> Result<Self, failure::Error> {
        Self::new(device, format, dim)
    }
}

impl TextureView<Dim2D,Depth> {
    pub fn depth_buffer_for(swapchain: &SwapChain) -> Result<Self, failure::Error> {
        Self::new(swapchain.device(), vk::Format::D32_SFLOAT, Dim2D::from(swapchain.extent()))
    }
}


pub struct StageTexture<D: Dim> {
    texture: TextureView<D, Color>,
    staging_buffer: OwnedBuffer<u8, Cpu>,
}

impl<D: Dim> StageTexture<D> {
    pub fn imageview(&self) -> &ImageView<Color> {
        self.texture.imageview()
    }
    pub fn texture(&self) -> &Texture<D, Color> {
        self.texture.texture()
    }
    pub fn take(self) -> TextureView<D, Color> {
        self.texture
    }
    pub fn staging_buffer(&self) -> &OwnedBuffer<u8, Cpu> {
        &self.staging_buffer
    }
}

impl StageTexture<Dim2D> {
    pub fn new(file: &Path, cmd_pool: &CommandPool, flip: bool) -> Result<Submitter<Self>, failure::Error> {
        let img = image::open(file).map_err(err_msg)?;
        let img = if flip { img.flipv() } else { img };
        let img = img.into_rgba8();
        let format = vk::Format::R8G8B8A8_SRGB;
        // let format: vk::Format = match img.color() {
        //     ColorType::Rgb8 => vk::Format::R8G8B8_UINT,
        //     ColorType::Rgba8 => vk::Format::R8G8B8A8_UINT,
        //     ColorType::Rgb16 => vk::Format::R16G16B16_UINT,
        //     ColorType::Rgba16 => vk::Format::R16G16B16A16_UINT,
        //     ColorType::Bgr8 => vk::Format::B8G8R8_UINT,
        //     ColorType::Bgra8 => vk::Format::B8G8R8A8_UINT,
        //     x => panic!("Invalid color scheme {:?} for image {:?}", x, file)
        // };
        let data = img.as_bytes();
        let staging_buffer = OwnedBuffer::<u8, Cpu>::new(cmd_pool.device(), data)?;
        let texture = TextureView::new(cmd_pool.device(), format, Dim2D::new(img.width(), img.height()))?;
        let mut slf = Submitter::new(Self { staging_buffer, texture },cmd_pool)?;
        let (cmd,tex) = slf.inner_val();
        cmd.cmd()
            .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?
            .layout_barrier(tex.texture(), vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .copy_to_image(tex.staging_buffer(), tex.texture(), vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .layout_barrier(tex.texture(), vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .end()?;
        cmd.submit()?;
        Ok(slf)
    }
}


