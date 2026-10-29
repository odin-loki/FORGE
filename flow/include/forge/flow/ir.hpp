#pragma once
// flow IR: a tensor graph treated as a program to compile once and replay.
#include <forge/spec/ops.gen.hpp>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace forge::flow {

using value_id = std::uint32_t;
inline constexpr value_id no_value = ~0u;

enum class node_kind : u8 { input, weight, constant, op, state_append, output };

struct node {
    node_kind kind = node_kind::op;
    spec::op op{};                    // for kind == op
    std::vector<value_id> inputs;
    spec::attrs attrs;
    shape_t shape;                    // filled by verify
    std::string name;                 // inputs/weights/state slots
    const float* weight = nullptr;    // kind == weight: external (e.g. mmap'd / decoded) data
    std::shared_ptr<tensor> constant; // kind == constant
    int device = 0;                   // filled by partition
};

class graph {
public:
    value_id input(std::string name, shape_t s) { return push({.kind = node_kind::input, .shape = std::move(s), .name = std::move(name)}); }
    value_id weight(std::string name, shape_t s, const float* data) {
        return push({.kind = node_kind::weight, .shape = std::move(s), .name = std::move(name), .weight = data});
    }
    value_id constant(tensor t) {
        node n{.kind = node_kind::constant, .shape = t.shape};
        n.constant = std::make_shared<tensor>(std::move(t));
        return push(std::move(n));
    }
    value_id scalar(float v) { return constant(tensor({1}, v)); }
    value_id apply(spec::op o, std::vector<value_id> in, spec::attrs a = {}) {
        return push({.kind = node_kind::op, .op = o, .inputs = std::move(in), .attrs = std::move(a)});
    }
    value_id apply(std::string_view opname, std::vector<value_id> in, spec::attrs a = {}) {
        auto o = spec::op_from_name(opname);
        if (!o) raise(errc::not_found, "unknown op '{}'", opname);
        return apply(*o, std::move(in), std::move(a));
    }
    // Stateful append along axis 0 to slot `slot`; returns the full history (state ++ x).
    value_id state_append(std::string slot, value_id x) {
        return push({.kind = node_kind::state_append, .inputs = {x}, .name = std::move(slot)});
    }
    void output(std::string name, value_id v) { outputs_.push_back({std::move(name), v}); }

    // fluent helpers
    value_id add(value_id a, value_id b) { return apply(spec::op::add, {a, b}); }
    value_id mul(value_id a, value_id b) { return apply(spec::op::mul, {a, b}); }
    value_id matmul(value_id a, value_id b) { return apply(spec::op::matmul, {a, b}); }

    std::vector<node>& nodes() { return nodes_; }
    const std::vector<node>& nodes() const { return nodes_; }
    node& at(value_id v) { return nodes_.at(v); }
    const node& at(value_id v) const { return nodes_.at(v); }
    std::vector<std::pair<std::string, value_id>>& outputs() { return outputs_; }
    const std::vector<std::pair<std::string, value_id>>& outputs() const { return outputs_; }
    std::size_t size() const { return nodes_.size(); }
private:
    value_id push(node n) { nodes_.push_back(std::move(n)); return value_id(nodes_.size() - 1); }
    std::vector<node> nodes_;
    std::vector<std::pair<std::string, value_id>> outputs_;
};

std::string to_string(const graph& g);

inline std::string to_string(const graph& g) {
    std::string s;
    for (std::size_t i = 0; i < g.size(); ++i) {
        const auto& n = g.at(value_id(i));
        s += "%" + std::to_string(i) + " " + shape_str(n.shape) + " = ";
        switch (n.kind) {
            case node_kind::input: s += "input " + n.name; break;
            case node_kind::weight: s += "weight " + n.name; break;
            case node_kind::constant: s += "const"; break;
            case node_kind::state_append: s += "state_append " + n.name; break;
            case node_kind::output: s += "output"; break;
            case node_kind::op: s += std::string(spec::info(n.op).name); break;
        }
        for (auto in : n.inputs) s += " %" + std::to_string(in);
        s += "\n";
    }
    for (auto& [nm, v] : g.outputs()) s += "output " + nm + " = %" + std::to_string(v) + "\n";
    return s;
}

} // namespace forge::flow
