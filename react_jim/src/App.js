import React, { useState } from 'react';
import './App.css';

function App() {
  const [tasks, setTasks] = useState([]);
  const [input, setInput] = useState('');

  const handleAddTask = () => {
    if (!input) return; 
    setTasks([...tasks, input]);
    setInput(''); 
  };

  const handleDeleteTask = (index) => {
    const newTasks = tasks.filter((task, taskIndex) => index !== taskIndex);
    setTasks(newTasks);
  };

  return (
    <div className="App">
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        type="text"
        placeholder="할 일을 입력하세요"
      />
      <button onClick={handleAddTask}>추가</button>
      <ul>
        {tasks.map((task, index) => (
          <li key={index}>
            {task} <button onClick={() => handleDeleteTask(index)}>삭제</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;
