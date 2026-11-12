import XCTest
@testable import Forge

/// Skips (XCTSkip) without FORGE_TEST_FPK; on a Mac/iPhone with a model it checks the round trip on Metal.
final class ForgeTests: XCTestCase {
    func testABI() { XCTAssertGreaterThanOrEqual(ForgeModel.abi, 4) }
    /// ABI 4 surface on shared weights. NEVER RUN (no Mac here); it mirrors tests/python_bindings.py.
    func testSharedWeightsAndSampling() throws {
        guard let p = ProcessInfo.processInfo.environment["FORGE_TEST_FPK"], FileManager.default.fileExists(atPath: p) else {
            throw XCTSkip("FORGE_TEST_FPK not set") }
        let w = try ForgeWeights(path: p, device: "cpu")
        XCTAssertEqual(w.device, "cpu")
        let a = try w.session(context: 64), b = try w.session()
        let ids = try a.tokenize("the weather")
        let ref = try a.generate(ids, maxNew: 8, options: GenerateOptions(ignoreEOS: true, reset: true))
        XCTAssertEqual(ref.count, 8); XCTAssertEqual(a.lastStopReason, .length)
        XCTAssertEqual(try b.generate(ids, maxNew: 8, options: GenerateOptions(ignoreEOS: true, reset: true)), ref)
        let hot = Sampling(temperature: 0.9, topK: 40, seed: 7, presencePenalty: 0.5)
        XCTAssertEqual(try a.generate(ids, maxNew: 8, sampling: hot, options: GenerateOptions(reset: true)),
                       try a.generate(ids, maxNew: 8, sampling: hot, options: GenerateOptions(reset: true)))
        let yn = try a.detokenize(a.generate(ids, maxNew: 8, sampling: hot, options: GenerateOptions(grammar: "root ::= \"yes\" | \"no\"", reset: true)))
        XCTAssertTrue(yn == "yes" || yn == "no"); XCTAssertEqual(a.lastStopReason, .grammar)
        var text = ""
        let streamed = try a.stream(ids, maxNew: 8, options: GenerateOptions(ignoreEOS: true, reset: true)) { text += $0; return false }
        XCTAssertEqual(streamed, ref); XCTAssertEqual(text, try a.detokenize(ref))
        a.reset(); _ = try a.forward(ids)
        let blob = try a.saveState(); try b.loadState(blob)
        XCTAssertEqual(b.position, Int64(ids.count)); XCTAssertEqual(try a.forward([ref[0]]), try b.forward([ref[0]]))
        XCTAssertEqual(try a.stopTokens(), [a.eosToken]); XCTAssertEqual(a.kind, "generative_sequence")
        let batch = try w.batch(maxActive: 2)
        XCTAssertEqual(try batch.generateMany([ids, ids, ids], maxNew: 8, sampling: Sampling(logitBias: [a.eosToken: -1e9])).count, 3)
        XCTAssertTrue(try Forge.jsonSchemaToGrammar("{\"enum\":[1,2]}").hasPrefix("root ::= "))
    }
    func testRoundTrip() throws {
        guard let p = ProcessInfo.processInfo.environment["FORGE_TEST_FPK"], FileManager.default.fileExists(atPath: p) else {
            throw XCTSkip("FORGE_TEST_FPK not set") }
        let m = try ForgeModel(path: p)
        XCTAssertEqual(try m.detokenize(try m.tokenize("hello", addBOS: false)), "hello")
        let out = try m.generate(try m.tokenize("hi"), maxNew: 4)
        XCTAssertTrue((1...4).contains(out.count))
        XCTAssertEqual(try m.forward([out[0]]).count, m.vocabSize)
    }
}
