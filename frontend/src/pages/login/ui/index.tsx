import type { FC } from 'react';
import LoginForm from './login-form';
import { Link } from 'react-router';

const LoginPage: FC = () => (
  <>
    <LoginForm />

    <div className='flex gap-2 items-center justify-center text-xs mt-4'>
      <p className='text-gray-400'>Нет аккаунта?</p>

      <Link to='/register' className='text-blue-400'>
        Зарегистрироваться
      </Link>
    </div>
  </>
);

export default LoginPage;
