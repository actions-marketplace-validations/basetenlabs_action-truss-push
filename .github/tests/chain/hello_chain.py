import random
import truss_chains as chains


class RandInt(chains.ChainletBase):
    async def run_remote(self, max_value: int) -> int:
        return random.randint(1, max_value)


@chains.mark_entrypoint
class HelloWorld(chains.ChainletBase):
    def __init__(self, rand_int=chains.depends(RandInt, retries=3)) -> None:
        self._rand_int = rand_int

    async def run_remote(self, max_value: int) -> str:
        num_repetitions = await self._rand_int.run_remote(max_value)
        return "Hello World! " * num_repetitions
