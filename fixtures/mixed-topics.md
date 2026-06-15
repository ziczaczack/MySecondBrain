# Mixed notebook — unrelated topics in one file

## Rust async runtime

Rust does not ship a built-in async runtime; instead the language provides
the async and await keywords and leaves execution to a library. Tokio is
the most widely used asynchronous runtime in the Rust ecosystem. When you
mark a function async, calling it returns a future that does nothing until
it is polled. The runtime's executor is what drives those futures to
completion, parking a task when it would block on I/O and waking it again
once the operating system signals readiness. You spawn concurrent work with
tokio::spawn, which hands a future to the scheduler so it can run on the
multi-threaded worker pool. The .await operator yields control back to the
executor at each suspension point instead of blocking the thread, which is
why a single runtime can juggle hundreds of thousands of in-flight tasks.
A common pitfall is calling blocking code inside an async task: it stalls
the worker thread and starves every other future scheduled on it, so you
move that work onto a dedicated blocking pool with spawn_blocking. Channels
let asynchronous tasks communicate without shared locks, and select lets a
task await several futures at once and act on whichever finishes first. The
combination of zero-cost futures, a work-stealing scheduler, and explicit
await points is what makes Rust's asynchronous runtime both fast and
predictable for high-concurrency network services and back-end systems.

## Sourdough bread baking

A completely different subject now. A sourdough starter is a living culture
of wild yeast and lactic acid bacteria kept alive by regular feedings of
flour and water. Hydration — the ratio of water to flour — controls how
slack or stiff the dough behaves on the bench. Bulk fermentation is where
flavour and strength build: the baker performs a series of stretch and
folds, lets the gluten relax, and watches for the dough to rise. Proofing
in a banneton overnight in the fridge slows the yeast and deepens the sour
tang. None of this has anything to do with software or concurrency.

## Roman aqueducts

Another unrelated topic. The aqueducts of ancient Rome carried water across
long distances using a remarkably gentle and consistent gradient. Engineers
surveyed routes so that water flowed downhill by gravity alone, sometimes
for tens of kilometres, through underground conduits and across towering
arched bridges. Settling tanks removed grit, and distribution basins split
the supply between public fountains, baths, and private homes.

## Photosynthesis

Finally, a biology note. Photosynthesis is the process by which plants,
algae, and some bacteria convert light energy into chemical energy stored
in sugars. Inside the chloroplast, chlorophyll absorbs photons and drives
the splitting of water, releasing oxygen as a by-product. This is
biochemistry, with no relationship to schedulers or futures.
