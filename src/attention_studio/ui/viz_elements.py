from PySide6.QtGui import QColor, QFont, QPen


class VizElement:
    def __init__(self, element_id, x, y, width, height, color, label, shape, layer_order=0, element_type="tensor"):
        self.id = element_id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.label = label
        self.shape = shape
        self.layer_order = layer_order
        self.element_type = element_type
        self._rect_item = None
        self._text_items = []

    @property
    def left(self):
        return self.x

    @property
    def top(self):
        return self.y

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    @property
    def center_x(self):
        return self.x + self.width // 2

    @property
    def center_y(self):
        return self.y + self.height // 2

    def apply_to_scene(self, scene):
        if self.shape == "rect":
            self._rect_item = scene.addRect(self.x, self.y, self.width, self.height)
            self._rect_item.setBrush(QColor(self.color.red(), self.color.green(), self.color.blue(), 50))
            self._rect_item.setPen(QPen(self.color, 1.5))
            self._rect_item.setZValue(10)
        elif self.shape == "circle":
            self._rect_item = scene.addEllipse(self.x, self.y, self.width, self.height)
            self._rect_item.setBrush(QColor(self.color.red(), self.color.green(), self.color.blue(), 70))
            self._rect_item.setPen(QPen(self.color, 2))
            self._rect_item.setZValue(15)

        if self.label:
            text = scene.addText(self.label[:12])
            text.setDefaultTextColor(QColor(220, 220, 230))
            text.setFont(QFont("SF Mono", 7))
            text.setPos(self.x + 2, self.y + 2)
            text.setZValue(20)
            self._text_items.append(text)


class VizConnection:
    def __init__(self, source_id, target_id, source_side="right", target_side="left", color=None):
        self.source_id = source_id
        self.target_id = target_id
        self.source_side = source_side
        self.target_side = target_side
        self.color = color or QColor(100, 100, 120)
        self._path_item = None
        self._arrow_item = None

    def apply_to_scene(self, scene):
        pass


def create_model_graph(num_layers, num_heads, hidden_size, head_dim, vocab_size, max_pos, scene):
    tw = 70
    th = 20
    ops = 22
    hs = 100

    elements = []
    connections = []

    elem_id = 0

    pre_layer_end_x = 60 + hs * 8

    e_input = VizElement(elem_id, 60, 100, tw, th, QColor(46, 204, 113), "input", "rect", 0, "tensor")
    elements.append(e_input)
    elem_id += 1

    e_wte = VizElement(elem_id, 60 + hs * 2, 70, tw, th, QColor(46, 204, 113), "wte", "rect", 1, "tensor")
    elements.append(e_wte)
    connections.append(VizConnection(e_input.id, e_wte.id, "right", "left"))
    elem_id += 1

    e_wpe = VizElement(elem_id, 60 + hs * 3, 180, tw, th, QColor(46, 204, 113), "wpe", "rect", 2, "tensor")
    elements.append(e_wpe)
    connections.append(VizConnection(e_input.id, e_wpe.id, "right", "left"))
    elem_id += 1

    e_add = VizElement(elem_id, 60 + hs * 4, 110, ops, ops, QColor(46, 204, 113), "+", "circle", 3, "operation")
    elements.append(e_add)
    connections.append(VizConnection(e_wte.id, e_add.id, "right", "left"))
    connections.append(VizConnection(e_wpe.id, e_add.id, "right", "left"))
    elem_id += 1

    e_hidden = VizElement(elem_id, 60 + hs * 5, 100, tw, th, QColor(46, 204, 113), "hidden", "rect", 4, "tensor")
    elements.append(e_hidden)
    connections.append(VizConnection(e_add.id, e_hidden.id, "right", "left"))
    elem_id += 1

    prev_layer_out = e_hidden

    for li in range(num_layers):
        layer_base = li * 15 * hs + pre_layer_end_x

        e_ln = VizElement(elem_id, layer_base, 100, tw, th, QColor(241, 196, 15), f"ln{li}", "rect", 5 + li * 15, "tensor")
        elements.append(e_ln)
        connections.append(VizConnection(prev_layer_out.id, e_ln.id, "right", "left"))
        elem_id += 1

        e_qkv = VizElement(elem_id, layer_base + hs * 2, 100, tw, th, QColor(52, 152, 219), f"qkv{li}", "rect", 6 + li * 15, "tensor")
        elements.append(e_qkv)
        connections.append(VizConnection(e_ln.id, e_qkv.id, "right", "left"))
        elem_id += 1

        e_q = VizElement(elem_id, layer_base + hs * 3, 50, tw, th, QColor(52, 152, 219), f"Q{li}", "rect", 7 + li * 15, "tensor")
        elements.append(e_q)
        connections.append(VizConnection(e_qkv.id, e_q.id, "right", "left"))
        elem_id += 1

        e_k = VizElement(elem_id, layer_base + hs * 4, 50, tw, th, QColor(52, 152, 219), f"K{li}", "rect", 8 + li * 15, "tensor")
        elements.append(e_k)
        connections.append(VizConnection(e_qkv.id, e_k.id, "right", "left"))
        elem_id += 1

        e_v = VizElement(elem_id, layer_base + hs * 5, 50, tw, th, QColor(52, 152, 219), f"V{li}", "rect", 9 + li * 15, "tensor")
        elements.append(e_v)
        connections.append(VizConnection(e_qkv.id, e_v.id, "right", "left"))
        elem_id += 1

        e_qk = VizElement(elem_id, layer_base + hs * 6, 75, ops, ops, QColor(155, 89, 182), "QK", "circle", 10 + li * 15, "operation")
        elements.append(e_qk)
        connections.append(VizConnection(e_q.id, e_qk.id, "right", "left"))
        connections.append(VizConnection(e_k.id, e_qk.id, "right", "left"))
        elem_id += 1

        e_sm = VizElement(elem_id, layer_base + hs * 7, 75, ops, ops, QColor(155, 89, 182), "sm", "circle", 11 + li * 15, "operation")
        elements.append(e_sm)
        connections.append(VizConnection(e_qk.id, e_sm.id, "right", "left"))
        elem_id += 1

        e_attn = VizElement(elem_id, layer_base + hs * 8, 75, ops, ops, QColor(155, 89, 182), "@V", "circle", 12 + li * 15, "operation")
        elements.append(e_attn)
        connections.append(VizConnection(e_sm.id, e_attn.id, "right", "left"))
        connections.append(VizConnection(e_v.id, e_attn.id, "right", "left"))
        elem_id += 1

        e_proj = VizElement(elem_id, layer_base + hs * 9, 100, tw, th, QColor(155, 89, 182), f"prj{li}", "rect", 13 + li * 15, "tensor")
        elements.append(e_proj)
        connections.append(VizConnection(e_attn.id, e_proj.id, "right", "left"))
        elem_id += 1

        e_add2 = VizElement(elem_id, layer_base + hs * 10, 102, ops, ops, QColor(46, 204, 113), "+", "circle", 14 + li * 15, "operation")
        elements.append(e_add2)
        connections.append(VizConnection(e_ln.id, e_add2.id, "right", "left"))
        connections.append(VizConnection(e_proj.id, e_add2.id, "right", "left"))
        elem_id += 1

        e_ln2 = VizElement(elem_id, layer_base + hs * 11, 100, tw, th, QColor(241, 196, 15), f"ln{li}", "rect", 15 + li * 15, "tensor")
        elements.append(e_ln2)
        connections.append(VizConnection(e_add2.id, e_ln2.id, "right", "left"))
        elem_id += 1

        e_fc1 = VizElement(elem_id, layer_base + hs * 12, 100, tw, th, QColor(230, 126, 34), f"fc1{li}", "rect", 16 + li * 15, "tensor")
        elements.append(e_fc1)
        connections.append(VizConnection(e_ln2.id, e_fc1.id, "right", "left"))
        elem_id += 1

        e_gelu = VizElement(elem_id, layer_base + hs * 13, 102, ops, ops, QColor(230, 126, 34), "GELU", "circle", 17 + li * 15, "operation")
        elements.append(e_gelu)
        connections.append(VizConnection(e_fc1.id, e_gelu.id, "right", "left"))
        elem_id += 1

        e_fc2 = VizElement(elem_id, layer_base + hs * 14, 100, tw, th, QColor(230, 126, 34), f"fc2{li}", "rect", 18 + li * 15, "tensor")
        elements.append(e_fc2)
        connections.append(VizConnection(e_gelu.id, e_fc2.id, "right", "left"))
        elem_id += 1

        e_add3 = VizElement(elem_id, layer_base + hs * 15, 102, ops, ops, QColor(46, 204, 113), "+", "circle", 19 + li * 15, "operation")
        elements.append(e_add3)
        connections.append(VizConnection(e_ln2.id, e_add3.id, "right", "left"))
        connections.append(VizConnection(e_fc2.id, e_add3.id, "right", "left"))
        elem_id += 1

        prev_layer_out = e_add3

    final_x = pre_layer_end_x + 17 * hs * num_layers + hs * 2

    e_ln_f = VizElement(elem_id, final_x, 100, tw, th, QColor(241, 196, 15), "ln_f", "rect", 100, "tensor")
    elements.append(e_ln_f)
    connections.append(VizConnection(prev_layer_out.id, e_ln_f.id, "right", "left"))
    elem_id += 1

    e_lm_head = VizElement(elem_id, final_x + hs, 100, tw, th, QColor(231, 76, 60), "lm_head", "rect", 101, "tensor")
    elements.append(e_lm_head)
    connections.append(VizConnection(e_ln_f.id, e_lm_head.id, "right", "left"))
    elem_id += 1

    e_logits = VizElement(elem_id, final_x + hs * 2, 100, tw, th, QColor(231, 76, 60), "logits", "rect", 102, "tensor")
    elements.append(e_logits)
    connections.append(VizConnection(e_lm_head.id, e_logits.id, "right", "left"))
    elem_id += 1

    e_sm_out = VizElement(elem_id, final_x + hs * 3, 102, ops, ops, QColor(231, 76, 60), "sm", "circle", 103, "operation")
    elements.append(e_sm_out)
    connections.append(VizConnection(e_logits.id, e_sm_out.id, "right", "left"))
    elem_id += 1

    return elements, connections
