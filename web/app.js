const state = { model: null };

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (k === "class") node.className = v;
    else if (k === "for") node.htmlFor = v;
    else node.setAttribute(k, v);
  });
  children.forEach(c => node.appendChild(c));
  return node;
}

function buildForm(model) {
  const fields = document.getElementById("fields");
  fields.innerHTML = "";

  model.attributes.forEach(attr => {
    const id = `field-${attr.name}`;
    const label = el("label", { for: id }, [document.createTextNode(attr.name)]);
    const select = el("select", { id, name: attr.name });

    attr.values.forEach(val => {
      const opt = el("option", { value: val }, [document.createTextNode(val)]);
      select.appendChild(opt);
    });

    const wrapper = el("div", {}, [label, select]);
    fields.appendChild(wrapper);
  });
}

function predict(input, model) {
  const scores = {};
  let bestClass = null;
  let bestLog = -Infinity;

  model.classes.forEach(cls => {
    let logp = Math.log(model.classPriors[cls] || model.epsilon);
    model.attributes.forEach(attr => {
      const val = input[attr.name];
      const idx = attr.values.indexOf(val);
      const probs = attr.condProbs[cls];
      const p = (idx >= 0 && probs && probs[idx] != null) ? probs[idx] : model.epsilon;
      logp += Math.log(p || model.epsilon);
    });
    scores[cls] = logp;
    if (logp > bestLog) {
      bestLog = logp;
      bestClass = cls;
    }
  });

  const maxLog = Math.max(...Object.values(scores));
  const expScores = {};
  let sum = 0;
  Object.entries(scores).forEach(([k, v]) => {
    const e = Math.exp(v - maxLog);
    expScores[k] = e;
    sum += e;
  });

  const probs = {};
  Object.entries(expScores).forEach(([k, v]) => {
    probs[k] = v / sum;
  });

  return { bestClass, probs };
}

async function init() {
  const result = document.getElementById("result");
  const probsBox = document.getElementById("probs");

  try {
    const res = await fetch("model.json");
    if (!res.ok) throw new Error("Failed to load model.json");
    const model = await res.json();
    state.model = model;
    buildForm(model);
    result.textContent = "Model loaded. Ready to predict.";
  } catch (err) {
    result.textContent = `Error: ${err.message}`;
  }

  const form = document.getElementById("predict-form");
  form.addEventListener("submit", evt => {
    evt.preventDefault();
    if (!state.model) return;

    const data = new FormData(form);
    const input = {};
    state.model.attributes.forEach(attr => {
      input[attr.name] = data.get(attr.name);
    });

    const { bestClass, probs } = predict(input, state.model);
    result.textContent = `Prediction: ${bestClass}`;
    probsBox.textContent = JSON.stringify(probs, null, 2);
  });
}

window.addEventListener("DOMContentLoaded", init);
