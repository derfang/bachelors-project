from manim import *
import numpy as np

# Helper functions
def joukowsky_map(z):
    if z == 0:
        return 0
    return z + 1 / z

def derivative(func, dt=1e-7):
    return lambda z: (func(z + dt) - func(z)) / dt

def cylinder_flow_vector_field(point, R=1, U=1):
    z = complex(point[0], point[1])
    return np.array([
        derivative(joukowsky_map)(z).real,
        derivative(joukowsky_map)(z).imag,
        0
    ])

def four_swirls_function(point):
    x, y = point[:2]
    result = (y**3 - 4 * y) * RIGHT + (x**3 - 16 * x) * UP
    result *= 0.05
    norm = np.linalg.norm(result)
    if norm == 0:
        return result
    return result

# Scenes
class Divergance(Scene):
    def construct(self):
        # Divergence
        def div_func(p):
            return p / 3
        div_vector_field = ArrowVectorField(div_func)
        stream_lines = StreamLines(div_func, stroke_width=2, max_anchors_per_line=10)
        stream_lines.shuffle()
        div_title = Text("Divergence").to_edge(UP)

        self.add(div_vector_field)
        self.play(
            AnimationGroup(*[Create(line) for line in stream_lines], lag_ratio=0.1),
            Write(div_title)
        )
        self.wait()

        # Curl
        def curl_func(p):
            return np.array([-p[1] / 3, p[0] / 3, 0])

        curl_vector_field = ArrowVectorField(curl_func)
        stream_lines = StreamLines(curl_func, stroke_width=2, max_anchors_per_line=30)
        stream_lines.shuffle()
        curl_title = Text("Curl").to_edge(UP)

        self.play(
            Transform(div_vector_field, curl_vector_field),
            Transform(div_title, curl_title),
        )
        self.play(AnimationGroup(*[Create(line) for line in stream_lines], lag_ratio=0.1))
        self.wait()

class TestVectorField(Scene):
    def construct(self):
        lines = StreamLines(
            four_swirls_function,
            stroke_width=2,
            max_anchors_per_line=30,
            virtual_time=3
        )
        self.add(lines)
        self.play(*[Create(line) for line in lines], run_time=5)
        self.wait(5)