use ash::vk;
use crate::device::Device;
use ash::version::DeviceV1_0;

pub struct ImageView{
    img:vk::ImageView,
    device:Device
}

impl ImageView{
    pub fn new(raw:vk::Image, format:vk::Format, device:&Device) -> Result<Self, ash::vk::Result> {
        let imageview_create_info = vk::ImageViewCreateInfo::builder()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image(raw);
        unsafe {device.device().create_image_view(&imageview_create_info, None)}.map(|img|Self{img,device:device.clone()})
    }
}

impl Drop for ImageView{
    fn drop(&mut self) {
        unsafe { self.device.device().destroy_image_view(self.img, None); }
    }
}