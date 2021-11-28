use ash::vk;

pub trait BufferType {
    const SHARING_MODE: vk::SharingMode;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags;
    const USAGE: vk::BufferUsageFlags;
}

pub trait CpuWriteable: BufferType {}

pub trait GpuWriteable: BufferType {}

pub trait DeviceLocal: BufferType {}

pub trait AsDescriptor: BufferType {}

pub trait AsStorage: AsDescriptor {}

impl AsDescriptor for Uniform{}

pub struct Uniform {}

impl BufferType for Uniform {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw() | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw());
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::UNIFORM_BUFFER;
}


impl CpuWriteable for Uniform {}

pub struct Gpu {}

impl DeviceLocal for Gpu{}

impl BufferType for Gpu {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(vk::BufferUsageFlags::VERTEX_BUFFER.as_raw() | vk::BufferUsageFlags::TRANSFER_DST.as_raw());
}

impl DeviceLocal for Storage{}

impl AsDescriptor for Storage{}

impl AsStorage for Storage{}

pub struct Storage {}

impl BufferType for Storage {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(vk::BufferUsageFlags::STORAGE_BUFFER.as_raw() | vk::BufferUsageFlags::VERTEX_BUFFER.as_raw() | vk::BufferUsageFlags::TRANSFER_DST.as_raw());
}


/**It's just like STORAGE buffer, but it does not have TRANSFER_DST flag, because it's meant to be initialised and used only on device*/
pub struct ProceduralStorage {}

impl BufferType for ProceduralStorage {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(vk::BufferUsageFlags::STORAGE_BUFFER.as_raw());
}
impl AsDescriptor for ProceduralStorage{}
impl AsStorage for ProceduralStorage{}
impl GpuWriteable for Storage {}
impl GpuWriteable for Gpu {}

pub struct GpuIndirect {}

impl BufferType for GpuIndirect {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw() | vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()  | vk::BufferUsageFlags::TRANSFER_DST.as_raw());
}

impl GpuWriteable for GpuIndirect {}
impl DeviceLocal for GpuIndirect {}
impl AsDescriptor for GpuIndirect{}
impl AsStorage for GpuIndirect {}

pub struct Cpu {}

impl BufferType for Cpu {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw() | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw());
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::TRANSFER_SRC;
}

impl CpuWriteable for Cpu {}
