import home from "../data/home.json";

export const PROFILE = {
  site: {
    SEO: {
      title: "Rohit Ravi – Machine Learning for Neuroscience",
      description:
        "Rohit Ravi works across machine learning, neuroscience, and computation. Explore research in neural decoding, biological modeling, and human genetics, and the personal motivation behind it.",
    },
  },
  timezone: "America/New_York",
  language: "en-US",
  firstName: "Rohit",
  name: "Rohit Ravi",
  headLine: home.headLine,
  headLine2: home.headLine2,
  website: "https://www.rohitravi.com",
  repo: "https://github.com/rohit-ravi2/personalwebsite",
  avatarUrl: "/avatar.png",
  introBio: home.bio,
  shortIntros: home.shortIntros,
  words: [
    "Arise, awake, and stop not till the goal is reached. — Swami Vivekananda",
  ],
  skills: [
    "Python · PyTorch · NumPy / SciPy · scikit-learn · LightGBM",
    "Deep learning: transformers, GNNs, CNNs, sequence models",
    "Neural decoding: sEMG, MEG, calcium imaging",
    "Statistical genetics: LDSC, Mendelian randomization, MAGMA",
    "Leakage audits, pre-registration, held-out evaluation",
    "In-browser ML demos: ONNX Runtime, Three.js, React",
    "Linux · Git · CUDA · Jupyter",
  ],
  links: {
    github: "https://github.com/rohit-ravi2",
    linkedin: "https://www.linkedin.com/in/rohit-ravi2/",
    orcid: "https://orcid.org/0009-0007-7568-0169",
    email: "mailto:rohit.ravi2@gmail.com",
    instagram: "https://instagram.com/rohitravi_",
  },
  studies: [
    {
      category: "Education",
      title: "B.A. Data Science (Philosophy minor)",
      institution: "New York University",
    },
  ],
};
