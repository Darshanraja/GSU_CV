// Add future modules here.
// Example:
// { name: "Module 3", page: "module3.html" }

const modules = [
  { name: "Module 2", page: "module2.html" },
  { name: "Module 3", page: "module3.html" },
  { name: "Module 4", page: "module4.html" },
];

const moduleSelect = document.getElementById("moduleSelect");
const openModuleButton = document.getElementById("openModuleButton");
const message = document.getElementById("message");

modules.forEach((module) => {
  const option = document.createElement("option");
  option.value = module.page;
  option.textContent = module.name;
  moduleSelect.appendChild(option);
});

openModuleButton.addEventListener("click", () => {
  const selectedPage = moduleSelect.value;

  if (!selectedPage) {
    message.textContent = "Please select a module.";
    return;
  }

  window.location.href = selectedPage;
});
