from pydantic import BaseModel
from typing import Literal

class Room(BaseModel):
    name: str
    area_m2: float
    zone: Literal["public", "private", "service", "work"]

class Link(BaseModel):
    a: str
    b: str
    strength: Literal["strong", "weak"]

class Program(BaseModel):
    rooms: list[Room]
    adjacencies: list[Link]


import ollama

SYSTEM = """You are an architectural programmer. Given a brief, list the rooms
with realistic areas in m², a zone for each, and which rooms should be adjacent.
Use the exact room names in the adjacencies. Include circulation (hall/corridor).
Every room must connect to at least one other room, and the hall must connect to the 
living room, so the diagram is one connected graph."""

def get_program(brief: str) -> Program:
    r = ollama.chat(
        model="qwen2.5:14b",
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": brief}],
        format=Program.model_json_schema(),
        options={"temperature": 0},
    )
    return Program.model_validate_json(r.message.content)

import networkx as nx
import matplotlib.pyplot as plt


# Step 3: drop any links to rooms that don't exist
def clean(p: Program) -> Program:
    names = {r.name for r in p.rooms}
    p.adjacencies = [l for l in p.adjacencies
                     if l.a in names and l.b in names and l.a != l.b]
    return p


# Step 4: build the graph
def to_graph(p: Program) -> nx.Graph:
    G = nx.Graph()
    for r in p.rooms:
        G.add_node(r.name, area=r.area_m2, zone=r.zone)
    for l in p.adjacencies:
        G.add_edge(l.a, l.b, weight=3 if l.strength == "strong" else 1)
    return G


# Step 5: draw it
COLORS = {"public": "#F4A261", "private": "#2A9D8F",
          "service": "#B8B8B8", "work": "#E76F51"}

def draw(G, brief, out="diagram.png"):
    pos = nx.spring_layout(G, weight="weight", seed=42, k=1.2)
    sizes = [G.nodes[n]["area"] * 300 for n in G]
    colors = [COLORS[G.nodes[n]["zone"]] for n in G]
    strong = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 3]
    weak = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 1]

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, edgelist=strong, width=2.5, edge_color="#333")
    nx.draw_networkx_edges(G, pos, edgelist=weak, width=1, style="dashed", edge_color="#999")
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, alpha=0.9)
    labels = {n: f"{n}\n{G.nodes[n]['area']:.0f} m²" for n in G}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    ax.set_title(brief, fontsize=12)
    ax.axis("off")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")


# Step 6: run it
if __name__ == "__main__":
    brief = input("Brief: ")
    draw(to_graph(clean(get_program(brief))), brief)
    print("saved diagram.png")