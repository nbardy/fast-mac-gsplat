import Metal
import simd

// Host-side skeleton for STGaussianRaster.metal.
// This is intended as a build guide, not a drop-in finished renderer.

struct ProjectedGaussianCPU {
    var mean_logAmp: SIMD4<Float>
    var H0: SIMD4<Float>
    var H1: SIMD4<Float>
    var H2: SIMD4<Float>
    var covDiag_depth: SIMD4<Float>
    var color_flags: SIMD4<Float>
}

struct RasterConfigCPU {
    var width: UInt32
    var height: UInt32
    var frameCount: UInt32
    var primitiveCount: UInt32

    var tileW: UInt32
    var tileH: UInt32
    var tilesX: UInt32
    var tilesY: UInt32

    var timeBins: UInt32
    var maxRefs: UInt32
    var time0: Float
    var time1: Float
    var truncR: Float
    var compositingMode: UInt32
}

final class STGaussianRasterizer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    let clearPSO: MTLComputePipelineState
    let countPSO: MTLComputePipelineState
    let fillPSO: MTLComputePipelineState
    let renderPSO: MTLComputePipelineState

    init(device: MTLDevice) throws {
        self.device = device
        guard let q = device.makeCommandQueue() else { throw NSError(domain: "Metal", code: 1) }
        self.queue = q
        self.library = try device.makeDefaultLibrary(bundle: .main)
        self.clearPSO = try device.makeComputePipelineState(function: library.makeFunction(name: "clear_uints")!)
        self.countPSO = try device.makeComputePipelineState(function: library.makeFunction(name: "count_cell_overlaps")!)
        self.fillPSO = try device.makeComputePipelineState(function: library.makeFunction(name: "fill_cell_overlaps")!)
        self.renderPSO = try device.makeComputePipelineState(function: library.makeFunction(name: "render_order_independent")!)
    }

    func makeBuffer<T>(_ values: [T], options: MTLResourceOptions = [.storageModeShared]) -> MTLBuffer {
        let byteCount = max(1, values.count * MemoryLayout<T>.stride)
        let b = device.makeBuffer(length: byteCount, options: options)!
        values.withUnsafeBytes { raw in
            if let base = raw.baseAddress, raw.count > 0 {
                b.contents().copyMemory(from: base, byteCount: raw.count)
            }
        }
        return b
    }

    func makeEmptyBuffer(byteCount: Int, options: MTLResourceOptions = [.storageModeShared]) -> MTLBuffer {
        return device.makeBuffer(length: max(1, byteCount), options: options)!
    }

    private func dispatch1D(_ enc: MTLComputeCommandEncoder, _ pso: MTLComputePipelineState, count: Int) {
        enc.setComputePipelineState(pso)
        let tg = MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: tg)
    }

    private func dispatch3D(_ enc: MTLComputeCommandEncoder, _ pso: MTLComputePipelineState, width: Int, height: Int, depth: Int) {
        enc.setComputePipelineState(pso)
        let tg = MTLSize(width: 8, height: 8, depth: 1)
        enc.dispatchThreads(MTLSize(width: width, height: height, depth: depth), threadsPerThreadgroup: tg)
    }

    func render(projected: [ProjectedGaussianCPU], cfg inputCfg: RasterConfigCPU) throws -> MTLBuffer {
        var cfg = inputCfg
        let numCells = Int(cfg.tilesX * cfg.tilesY * cfg.timeBins)
        let pgsBuffer = makeBuffer(projected)
        let cellCounts = makeEmptyBuffer(byteCount: numCells * MemoryLayout<UInt32>.stride)
        let cellWriteHeads = makeEmptyBuffer(byteCount: numCells * MemoryLayout<UInt32>.stride)
        let overflow = makeEmptyBuffer(byteCount: MemoryLayout<UInt32>.stride)

        // 1. Clear cellCounts.
        do {
            var clearCount = UInt32(numCells)
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setBuffer(cellCounts, offset: 0, index: 0)
            enc.setBytes(&clearCount, length: MemoryLayout<UInt32>.stride, index: 1)
            dispatch1D(enc, clearPSO, count: numCells)
            enc.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
        }

        // 2. Count overlaps.
        do {
            var cfgCopy = cfg
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setBuffer(pgsBuffer, offset: 0, index: 0)
            enc.setBuffer(cellCounts, offset: 0, index: 1)
            enc.setBytes(&cfgCopy, length: MemoryLayout<RasterConfigCPU>.stride, index: 2)
            dispatch1D(enc, countPSO, count: projected.count)
            enc.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
        }

        // 3. CPU exclusive scan prototype. Replace with GPU scan for production.
        let countsPtr = cellCounts.contents().bindMemory(to: UInt32.self, capacity: numCells)
        var offsets = Array(repeating: UInt32(0), count: numCells + 1)
        for i in 0..<numCells {
            offsets[i + 1] = offsets[i] + countsPtr[i]
        }
        let totalRefs = Int(offsets[numCells])
        cfg.maxRefs = UInt32(totalRefs)
        let offsetsBuffer = makeBuffer(offsets)
        let indicesBuffer = makeEmptyBuffer(byteCount: max(1, totalRefs) * MemoryLayout<UInt32>.stride)

        // 4. Clear write heads and overflow.
        do {
            var clearCount = UInt32(numCells)
            var one = UInt32(1)
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setBuffer(cellWriteHeads, offset: 0, index: 0)
            enc.setBytes(&clearCount, length: MemoryLayout<UInt32>.stride, index: 1)
            dispatch1D(enc, clearPSO, count: numCells)
            enc.setBuffer(overflow, offset: 0, index: 0)
            enc.setBytes(&one, length: MemoryLayout<UInt32>.stride, index: 1)
            dispatch1D(enc, clearPSO, count: 1)
            enc.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
        }

        // 5. Fill overlap lists.
        do {
            var cfgCopy = cfg
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setBuffer(pgsBuffer, offset: 0, index: 0)
            enc.setBuffer(offsetsBuffer, offset: 0, index: 1)
            enc.setBuffer(cellWriteHeads, offset: 0, index: 2)
            enc.setBuffer(indicesBuffer, offset: 0, index: 3)
            enc.setBuffer(overflow, offset: 0, index: 4)
            enc.setBytes(&cfgCopy, length: MemoryLayout<RasterConfigCPU>.stride, index: 5)
            dispatch1D(enc, fillPSO, count: projected.count)
            enc.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
        }

        // 6. Render all frames into one linear float4 buffer.
        let outCount = Int(cfg.width * cfg.height * cfg.frameCount)
        let outBuffer = makeEmptyBuffer(byteCount: outCount * MemoryLayout<SIMD4<Float>>.stride)
        do {
            var cfgCopy = cfg
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setBuffer(pgsBuffer, offset: 0, index: 0)
            enc.setBuffer(offsetsBuffer, offset: 0, index: 1)
            enc.setBuffer(cellCounts, offset: 0, index: 2)
            enc.setBuffer(indicesBuffer, offset: 0, index: 3)
            enc.setBuffer(outBuffer, offset: 0, index: 4)
            enc.setBytes(&cfgCopy, length: MemoryLayout<RasterConfigCPU>.stride, index: 5)
            dispatch3D(enc, renderPSO, width: Int(cfg.width), height: Int(cfg.height), depth: Int(cfg.frameCount))
            enc.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
        }

        return outBuffer
    }
}
