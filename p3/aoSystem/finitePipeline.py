"""Bounded CPU preparation ahead of serial device propagation."""
from concurrent.futures import ThreadPoolExecutor


class DirectionalPreparation:
    """One CPU producer and at most two batches of complete science weights.

    All device calls stay on the prediction thread. Only the producer queries
    the backend's directional prior; reconstruction inputs must stay immutable
    until exit. Results are consumed in the requested field order. Exceptions
    cancel queued work and join the producer before releasing the backend.
    No PSFs, covariance terms or reconstruction components are cached or reduced.
    """
    def __init__(self,backend,fields,batch):
        self.backend=backend
        self.fields=tuple(fields)
        if len(set(self.fields))!=len(self.fields):
            raise ValueError('Directional preparation requires unique fields')
        if not isinstance(batch,int) or batch<1:
            raise ValueError('Directional batch must be a positive integer')
        self.batch=batch
        self.ready={}
        self.next=0
        self.executor=None
        self.future=None

    def _submit(self):
        fields=self.fields[self.next:self.next+self.batch]
        self.next+=len(fields)
        self.future=self.executor.submit(self.backend.prepare_science_rows,fields) if fields else None

    def __enter__(self):
        if getattr(self.backend,'_directional_preparation',None) is not None:
            raise RuntimeError('Directional preparation is already active')
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='p3-science')
        self.backend._directional_preparation=self
        try:self._submit()
        except BaseException:
            self.__exit__()
            raise
        return self

    def take(self,field):
        if not self.ready:
            if self.future is None:raise ValueError('No directional preparation remains')
            self.ready=self.future.result()
            self._submit()
        if next(iter(self.ready))!=field:
            raise ValueError('Consume directional preparation in requested field order')
        return self.ready.pop(field)

    def __exit__(self,*exc):
        try:
            if self.future is not None:self.future.cancel()
            self.executor.shutdown(wait=True,cancel_futures=True)
        finally:
            self.ready.clear();self.future=None
            self.backend._directional_preparation=None
