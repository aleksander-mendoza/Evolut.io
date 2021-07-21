use crate::fence::Fence;
use ash::vk;

pub struct GpuFuture<T>{
    fence:Fence,
    val:T
}

pub enum FutureOutput<T>{
    Timeout(GpuFuture<T>),
    Error(vk::Result),
    Success(Fence, T)
}
impl <T> GpuFuture<T>{
    pub fn new(val:T,fence:Fence)->Self{
        Self{val,fence}
    }
    pub fn get(self)->Result<(Fence,T),vk::Result>{
        self.fence.wait(None).map(move |()|{
            let Self{ fence, val } = self;
            (fence,val)
        })
    }
    pub fn get_in_time(self,timeout:u64)->FutureOutput<T>{
        match self.fence.wait(Some(timeout)){
            Ok(()) => {
                let Self{ fence, val } = self;
                FutureOutput::Success(fence,val)
            },
            Err(vk::Result::TIMEOUT) => FutureOutput::Timeout(self),
            Err(err) => FutureOutput::Error(err)
        }
    }
}