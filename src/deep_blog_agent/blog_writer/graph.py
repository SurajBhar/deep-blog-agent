"""Graph construction for the blog writer workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from deep_blog_agent.blog_writer.contracts import BlogWorkflowState
from deep_blog_agent.blog_writer.nodes import BlogWriterNodes, route_next
from deep_blog_agent.providers.factory import ProviderBundle


def build_graph(providers: ProviderBundle):
    """Compile the blog writer graph using injected providers."""
    nodes = BlogWriterNodes(providers=providers)

    reducer_graph = StateGraph(BlogWorkflowState)
    reducer_graph.add_node("merge_content", nodes.merge_content)
    reducer_graph.add_node("decide_images", nodes.decide_images)
    reducer_graph.add_node("generate_and_place_images", nodes.generate_and_place_images)
    reducer_graph.add_edge(START, "merge_content")
    reducer_graph.add_edge("merge_content", "decide_images")
    reducer_graph.add_edge("decide_images", "generate_and_place_images")
    reducer_graph.add_edge("generate_and_place_images", END)
    reducer_subgraph = reducer_graph.compile()

    graph = StateGraph(BlogWorkflowState)
    graph.add_node("router", nodes.router_node)
    graph.add_node("research", nodes.research_node)
    graph.add_node("orchestrator", nodes.orchestrator_node)
    graph.add_node("worker", nodes.worker_node)
    graph.add_node("reducer", reducer_subgraph)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_next,
        {"research": "research", "orchestrator": "orchestrator"},
    )
    graph.add_edge("research", "orchestrator")
    graph.add_conditional_edges("orchestrator", nodes.fanout, ["worker"])
    graph.add_edge("worker", "reducer")
    graph.add_edge("reducer", END)

    return graph.compile()
